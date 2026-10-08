# continuation_gateway

English | [简体中文](README.zh-CN.md)

> [!IMPORTANT]
> **Only SGLang is supported as the inference engine at present.** vLLM is not supported
> yet: the continuation request relies on the `continue_final_message` /
> `add_generation_prompt=false` semantics, which SGLang implements uniformly, while vLLM
> leaves them to each model's chat template / tokenizer, so many models ignore them or
> cannot continue from a partial reply.

A crash-continuation gateway for LLM inference services. It sits between clients and an
OpenAI-compatible inference service (currently SGLang only — see the note above).

An inference instance can crash in the middle of a streaming generation, have its connection
reset, or stall (the process is alive but stops emitting tokens). The client is then left with
a truncated reply that never received a `finish_reason`, and usually has no choice but to retry
the whole request — the content already generated and the tokens already spent are wasted.
When this gateway detects that a streaming response has stalled (idle timeout) or has been
disconnected, it automatically issues **one** continuation request: the reasoning/content
already sent to the client is assembled into an assistant prefix, and the model is asked to
carry on from there. To the client, it is still a single, continuous, complete streaming
response.

The way the prefix is assembled is dispatched by model (see
`continuation_gateway/reconstruct.py`). Two formats are built in: Kimi-K3 (XTML) and a generic
`<think>...</think>` format. Supporting a model with another format only takes adding a
builder there.

```
                               leg1: original request
  client ──▶ continuation_gateway ──────────────────▶ downstream router ──▶ inference instance
     ▲         │  forwards chunk by chunk, parses SSE   (SGLang)
     │         │  accumulates reasoning / content
     │         │  detects stall / disconnect (no finish_reason seen)
     │         ▼
     │      builds the prefix, issues leg2: continuation request ──▶ downstream router
     │         │                                    (can be set separately via CONTINUATION_URL)
     └─────────┘  leg2's output continues right after what leg1 already forwarded;
                  id / reasoning field name / usage are rewritten before forwarding
```

## Design principles

- **Orchestration at the gateway layer; the inference engine is untouched.** Continuation
  requires parsing SSE semantics, remembering what has already been forwarded, issuing an
  extra request on its own initiative, and rewriting `usage` in place. So within a single
  request the gateway is stateful rather than a transparent byte-forwarder.
- **Conservative first: when in doubt, pass through.** A request that is out of scope, or
  does not meet the conditions for continuation, behaves exactly as if this gateway were not
  there: it does not hang up the connection, swallow exceptions, or rewrite the response.
  The gateway steps in only when a meaningful continuation can actually be made.
- **One continuation, at most two legs.** A single client request involves at most leg1 +
  leg2. If leg2 itself fails, no third leg is issued and the request ends as it stands — no
  worse than having no gateway at all. This is also why the `usage` correction formula can
  assume "exactly two legs".
- **The gateway must not become a point of failure itself.** Any unexpected error during the
  continuation phase is only logged, never propagated; content already forwarded to the client
  is unaffected.
- **Transparent to the client.** The `id` and the reasoning field name stay consistent across
  the whole stream. The recovered portion is counted in this request's `usage`, and the
  continuation request's `max_tokens` is reduced by what has already been generated, so the
  original budget is never exceeded.
- **Billing is addition-only and does not aim for absolute precision.** The recovered content
  is converted to a token count by character-based estimation and added to the usage reported
  by leg2. A small deviation is accepted in preference to the risk of a completely wrong
  measurement, and nothing depends on the downstream's `/v1/tokenize`.
- **No cross-request state.** All state lives in the coroutine of a single request. Replicas
  share nothing, so the gateway scales horizontally and needs no sticky sessions.

Out of scope for this gateway: general request retry (any problem before the first chunk
arrives is propagated as-is), choosing or avoiding failed inference instances (left to the
downstream router), and admission control and rate limiting (left to the upstream ingress).

## Scope

Only free-text scenarios are handled, where **no tool_call chunk had appeared at the time of
the crash and the request is not constrained by `response_format` to `json_object` /
`json_schema`**. The following cases are passed through without continuation; see the comments
on `should_intervene()` in `continuation_gateway/server.py` for details:

- A crash after a tool_call chunk has already appeared: under the standard
  `continue_final_message` semantics there is no "tool call still being generated" state, so
  supporting it would require changing the SGLang source. `BUFFER_TOOL_CALLS=true` (see below)
  works around this limitation: tool_call chunks are held in gateway memory and released to
  the client only once complete; on a crash the whole held batch is discarded and the
  continuation starts from the state before the tool_call appeared. The client never received
  a tool_call, so there is no "cannot be continued" state.
- Structured output with `response_format` set to `json_object` / `json_schema`: the
  constrained JSON content is mixed into ordinary content chunks, and there is no separate
  signal to recognize it.
- Requests that already carry `continue_final_message` from the client: continuation is not
  stacked on top.
- Request bodies larger than `MAX_CONTINUATION_BODY_MB` (by default, large multimodal bodies).

## Building the image

The repository root includes a `Dockerfile` (the only dependency is `aiohttp`). The base image
and the PyPI index are build args:

| build arg | default | description |
| --- | --- | --- |
| `BUILDER_IMAGE` | `python:3.12-slim` | Base image used to build and run |
| `PIP_INDEX_URL` | empty (direct PyPI) | Optional PyPI mirror |

```bash
docker build -t continuation-gateway:latest .

# If Docker Hub / PyPI are not reachable, use your own registry and PyPI mirror:
docker build -t continuation-gateway:latest \
    --build-arg BUILDER_IMAGE=<registry>/python:3.12-slim \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

Run the container (environment variables are described in the next section):

```bash
docker run --rm -p 8000:8000 \
    -e DOWNSTREAM_URL=http://<router>:<port> \
    -e CONTINUATION_MODELS=<model name> \
    continuation-gateway:latest
```

## Running

Without a container, run it directly with Python 3.12 (`pip install -r requirements.txt`).

The downstream is expected to be a datacenter/cluster router (not a specific SGLang instance
directly). The original request and the continuation request go to the same URL by default:

```bash
DOWNSTREAM_URL=http://<router>:<port> CONTINUATION_MODELS=<model name> \
    python -m continuation_gateway.server
```

Optional environment variables (defaults are in `continuation_gateway/config.py`): `PORT`,
`CONTINUATION_URL` (send the continuation request to a different address; if unset it is the
same as `DOWNSTREAM_URL`), `CONTINUATION_ENABLED` (default `true`; when set to `false` the
gateway still does all of leg1's monitoring, TRIGGERED detection and logging, but does not
actually issue the second leg — useful for verifying the monitoring path itself without adding
real continuation load to the downstream), `STALL_IDLE_TIMEOUT_SECONDS`,
`CONNECT_TIMEOUT_SECONDS`, `MAX_CONTINUATION_BODY_MB`, `MAX_REQUEST_BODY_MB`,
`CJK_CHARS_PER_TOKEN`, `OTHER_CHARS_PER_TOKEN`.

`BUFFER_TOOL_CALLS` (default `false`; behavior is unchanged when off; it only takes effect
when `CONTINUATION_ENABLED=true`, see below): once a chunk carrying `tool_calls` appears on
leg1, that chunk and every line after it are held in gateway memory until `finish_reason` (or
`[DONE]`) arrives, and are then forwarded all at once. If the stream stalls or disconnects
while holding, the held content is discarded as a whole, and the continuation starts from the
thinking-partial / content-done state before the first tool_call chunk (the continuation leg
regenerates, and may produce a tool_call again). When there is no recoverable content, the
held content is released to the client unchanged. Cost: tool_call arguments no longer arrive
chunk by chunk but all at once after the tool_call completes.

`CONTINUATION_ENABLED=true` is a prerequisite for `BUFFER_TOOL_CALLS` to take effect: the only
purpose of holding is to keep continuation feasible. Once the continuation itself is turned
off, holding would buy no recovery and would only make the tool_call arguments reach the client
a round later for nothing, so with this combination the gateway skips holding and treats the
request as a plain pass-through, the same as when `BUFFER_TOOL_CALLS` is not enabled.

## Tests

`tests/smoke_test.py`: does not connect to a real cluster. It starts a local fake downstream
(simulating SGLang's streaming responses) plus the real `continuation_gateway` service, and
uses real HTTP requests to verify the gateway's own forwarding and orchestration logic
(continuation triggered by stall/disconnect, scenarios such as tool_call / response_format /
oversized body being left alone, usage rewriting, and prefix reconstruction dispatched by
model). It does not verify whether the continued content is semantically coherent — that has
to be verified on a real cluster.

```bash
python tests/smoke_test.py
```

## License

This project is open-sourced under the [Apache License 2.0](LICENSE).
