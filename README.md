# continuation_gateway

Kimi-K3 崩溃续写网关：流式响应中途卡住（idle timeout）或断连时，自动发起一次续写请求，
把已经吐给客户端的 reasoning/content 拼成 prefix 续着生成，尽量把内容补完整。

## 范围（v1）

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

## 运行

下游预期是一个机房/集群路由网关（不是直连某个具体 SGLang 实例），原始请求和续写请求默认打
同一个 URL：

```bash
DOWNSTREAM_URL=http://<路由网关>:<port> CONTINUATION_MODELS=<Kimi-K3 的 model 名字> \
    python -m continuation_gateway.server
```

可选环境变量（默认值见 `continuation_gateway/config.py`）：`PORT`、`CONTINUATION_URL`
（续写请求单独打去另一个地址，不设置就跟 `DOWNSTREAM_URL` 一样）、`CONTINUATION_ENABLED`
（默认 `true`；设成 `false` 时网关仍然完整做 leg1 的监测/TRIGGERED 判定和留痕，只是不真的
发第二条腿，用于验证监测链路本身、不给下游增加真实续写负载）、`STALL_IDLE_TIMEOUT_SECONDS`、
`CONNECT_TIMEOUT_SECONDS`、`MAX_CONTINUATION_BODY_MB`、`MAX_REQUEST_BODY_MB`、
`CJK_CHARS_PER_TOKEN`、`OTHER_CHARS_PER_TOKEN`。

`BUFFER_TOOL_CALLS`（默认 `false`，关闭时行为不变）：leg1 上出现带 `tool_calls` 的 chunk 后，
这一个 chunk 和之后的所有行都暂存在网关内存里，直到收到 `finish_reason`（或 `[DONE]`）再一次性
转发。暂存期间卡住/断连时，暂存内容整批丢弃，按第一个 tool_call chunk 之前的
thinking-partial/content-done 状态续写（续写腿重新生成，可能又是一个 tool_call）。没有可恢复
内容、或 `CONTINUATION_ENABLED=false` 等不会真的续写的情况，暂存内容原样放给客户端。代价：
tool_call 参数不再逐 chunk 流式到达，而是在 tool_call 完成后一次性到达。

## 测试

`tests/smoke_test.py`：不连真实集群，起一个本地假 downstream（模拟 SGLang 的流式响应）+
真的 `continuation_gateway` 服务，用真实 HTTP 请求验证网关自己的转发/编排逻辑（卡住/断连
触发续写、tool_call/response_format/超限 body 等场景老实不救、usage 改写、按 model 分派的
前缀重建）。不验证续写内容语义是否连贯——那部分要在真实集群上验证。

```bash
python tests/smoke_test.py
```
