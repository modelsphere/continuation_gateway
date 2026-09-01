# continuation_gateway

Kimi-K3 崩溃续写网关：流式响应中途卡住（idle timeout）或断连时，自动发起一次续写请求，
把已经吐给客户端的 reasoning/content 拼成 prefix 续着生成，尽量把内容补完整。

## 范围（v1）

只处理"崩溃时还没出现过 tool_call chunk、且请求本身没有用 `response_format` 约束成
`json_object`/`json_schema`"的自由文本场景。以下情况直接透传、不介入续写，详见
`continuation_gateway/server.py` 里 `should_intervene()` 的注释：

- 已经出现过 tool_call chunk 才崩的：`continue_final_message` 标准语义下没有"tool 还在
  生成中"的状态位，要支持需要改 SGLang 源码。
- `response_format` 是 `json_object`/`json_schema` 的结构化输出：约束到的 JSON 内容混在
  普通 content chunk 里，没有独立信号能识别。
- 客户端自己发的 `continue_final_message` 请求：不再叠加续写。
- 请求体超过 `MAX_CONTINUATION_BODY_MB`（默认多模态大 body）。

## 运行

下游预期是一个机房/集群路由网关（不是直连某个具体 SGLang 实例），原始请求和续写请求打
同一个 URL：

```bash
DOWNSTREAM_URL=http://<路由网关>:<port> CONTINUATION_MODELS=<Kimi-K3 的 model 名字> \
    python -m continuation_gateway.server
```

可选环境变量（默认值见 `continuation_gateway/config.py`）：`TOKENIZE_URL`、`PORT`、
`STALL_IDLE_TIMEOUT_SECONDS`、`CONNECT_TIMEOUT_SECONDS`、`MAX_CONTINUATION_BODY_MB`。
