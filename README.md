# continuation_gateway

LLM 推理服务的崩溃续写网关：位于客户端和 OpenAI 兼容的推理服务（如 SGLang、vLLM）之间。

推理实例在流式生成中途可能崩溃、连接被重置，或者卡住（进程没死但不再吐 token）。客户端
这时拿到的是一段被截断、没有 `finish_reason` 的回复，通常只能整体重试，已经生成的内容和
已经消耗的 token 都白费了。本网关在检测到流式响应中途卡住（idle timeout）或断连时，自动
发起**一次**续写请求：把已经吐给客户端的 reasoning/content 拼成 assistant prefix，让模型
接着往下生成，客户端看到的仍然是一条连续、完整的流式响应。

前缀拼接方式按 model 分派（见 `continuation_gateway/reconstruct.py`），内置 Kimi-K3（XTML）
和通用 `<think>...</think>` 两种格式，接入其它格式的模型只需要在那里加一个 builder。

```
                                  leg1：原始请求
  client ──▶ continuation_gateway ─────────────────▶ 下游路由网关 ──▶ 推理实例
     ▲         │  逐 chunk 转发 + 解析 SSE              (SGLang / vLLM)
     │         │  累积 reasoning / content
     │         │  检测卡住 / 断连（没见过 finish_reason）
     │         ▼
     │      拼 prefix，发起 leg2：续写请求 ───────────▶ 下游路由网关
     │         │                                   （可用 CONTINUATION_URL 单独指定）
     └─────────┘  leg2 的输出接在 leg1 已转发内容之后，
                  改写 id / 思考字段名 / usage 后继续转发
```

## 设计思想

- **网关层编排，不改推理引擎。** 续写需要解析 SSE 语义、记住已经转发了什么、主动发起额外
  请求、原地改写 usage，所以网关在单个请求内是有状态的，而不是透明的字节转发。
- **保守优先，拿不准就透传。** 不在覆盖范围、或不满足续写条件的请求，行为与没有这层网关时
  完全一致：不主动挂断连接、不吞异常、不改写响应。只在确实能做一次有意义的续写时才介入。
- **单次续写，最多两条腿。** 单个客户端请求最多 leg1 + leg2。leg2 自己再失败时不发第三条腿，
  按现状收尾，效果上不比没有网关更差；usage 修正公式也因此可以假设"恰好两条腿"。
- **网关自身不能成为故障点。** 续写阶段的任何意外错误只记日志、不外抛，已经转发给客户端的
  内容不受影响。
- **对客户端无感。** 整条流里 `id`、思考字段名前后一致；续写救回的部分也计入本次请求的
  `usage`，续写请求的 `max_tokens` 扣减已经生成的部分，不会超出原预算。
- **计费只做加法，不追求绝对精确。** 被救回的内容按字符数估算 token，加到 leg2 真实上报的
  usage 上；宁可小幅偏差，也不引入"完全测错"的风险，也不依赖下游的 `/v1/tokenize`。
- **无跨请求状态。** 所有状态都在单个请求的协程里，副本之间不共享任何东西，可以水平扩展，
  不需要粘性会话。

不属于本网关的职责：通用请求重试（第一个 chunk 到达之前的任何问题都原样上抛）、选择/避开
故障推理实例（交给下游路由网关）、准入控制和限流（交给上游入口层）。

## 范围

只处理"崩溃时还没出现过 tool_call chunk、且请求本身没有用 `response_format` 约束成
`json_object`/`json_schema`"的自由文本场景。以下情况直接透传、不介入续写，详见
`continuation_gateway/server.py` 里 `should_intervene()` 的注释：

- 已经出现过 tool_call chunk 才崩的：`continue_final_message` 标准语义下没有"tool 还在
  生成中"的状态位，要支持需要改 SGLang 源码。可以用 `BUFFER_TOOL_CALLS=true`（见下）绕开
  这个限制：tool_call chunk 在网关内暂存、完整后才放给客户端，崩溃时整批丢弃并按 tool_call
  出现之前的状态续写，客户端从来没收到过 tool_call，也就不存在"续不上"的状态。
- `response_format` 是 `json_object`/`json_schema` 的结构化输出：约束到的 JSON 内容混在
  普通 content chunk 里，没有独立信号能识别。
- 客户端自己发的 `continue_final_message` 请求：不再叠加续写。
- 请求体超过 `MAX_CONTINUATION_BODY_MB`（默认多模态大 body）。

## 构建镜像

仓库根目录自带 `Dockerfile`（依赖只有 `aiohttp`）。基础镜像和 PyPI 源都是 build arg：

| build arg | 默认值 | 说明 |
| --- | --- | --- |
| `BUILDER_IMAGE` | `python:3.12-slim` | 构建并运行用的基础镜像 |
| `PIP_INDEX_URL` | 空（直连 PyPI） | 可选的 PyPI 镜像 |

```bash
docker build -t continuation-gateway:latest .

# 访问不了 Docker Hub / PyPI 时，换成自己的镜像仓库和 PyPI 镜像：
docker build -t continuation-gateway:latest \
    --build-arg BUILDER_IMAGE=<registry>/python:3.12-slim \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

运行容器（环境变量说明见下一节）：

```bash
docker run --rm -p 8000:8000 \
    -e DOWNSTREAM_URL=http://<路由网关>:<port> \
    -e CONTINUATION_MODELS=<model 名字> \
    continuation-gateway:latest
```

## 运行

不使用容器时，直接用 Python 3.12 跑（`pip install -r requirements.txt`）。

下游预期是一个机房/集群路由网关（不是直连某个具体 SGLang 实例），原始请求和续写请求默认打
同一个 URL：

```bash
DOWNSTREAM_URL=http://<路由网关>:<port> CONTINUATION_MODELS=<model 名字> \
    python -m continuation_gateway.server
```

可选环境变量（默认值见 `continuation_gateway/config.py`）：`PORT`、`CONTINUATION_URL`
（续写请求单独打去另一个地址，不设置就跟 `DOWNSTREAM_URL` 一样）、`CONTINUATION_ENABLED`
（默认 `true`；设成 `false` 时网关仍然完整做 leg1 的监测/TRIGGERED 判定和留痕，只是不真的
发第二条腿，用于验证监测链路本身、不给下游增加真实续写负载）、`STALL_IDLE_TIMEOUT_SECONDS`、
`CONNECT_TIMEOUT_SECONDS`、`MAX_CONTINUATION_BODY_MB`、`MAX_REQUEST_BODY_MB`、
`CJK_CHARS_PER_TOKEN`、`OTHER_CHARS_PER_TOKEN`。

`BUFFER_TOOL_CALLS`（默认 `false`，关闭时行为不变；只在 `CONTINUATION_ENABLED=true` 时才会
真的生效，见下）：leg1 上出现带 `tool_calls` 的 chunk 后，这一个 chunk 和之后的所有行都暂存在
网关内存里，直到收到 `finish_reason`（或 `[DONE]`）再一次性转发。暂存期间卡住/断连时，暂存
内容整批丢弃，按第一个 tool_call chunk 之前的 thinking-partial/content-done 状态续写（续写腿
重新生成，可能又是一个 tool_call）。没有可恢复内容的情况，暂存内容原样放给客户端。代价：
tool_call 参数不再逐 chunk 流式到达，而是在 tool_call 完成后一次性到达。

`CONTINUATION_ENABLED=false` 是 `BUFFER_TOOL_CALLS` 生效的大前提：暂存唯一的目的是保住续写
的可行性，续写这个动作本身被关掉之后暂存不会换来任何补救，只会让 tool_call 参数白白多等一轮
才到达客户端，所以这个组合下网关直接跳过暂存、按普通透传处理，跟没打开 `BUFFER_TOOL_CALLS`
时行为一致。

## 测试

`tests/smoke_test.py`：不连真实集群，起一个本地假 downstream（模拟 SGLang 的流式响应）+
真的 `continuation_gateway` 服务，用真实 HTTP 请求验证网关自己的转发/编排逻辑（卡住/断连
触发续写、tool_call/response_format/超限 body 等场景老实不救、usage 改写、按 model 分派的
前缀重建）。不验证续写内容语义是否连贯——那部分要在真实集群上验证。

```bash
python tests/smoke_test.py
```

## License

本项目以 [Apache License 2.0](LICENSE) 开源。
