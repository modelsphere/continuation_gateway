"""vLLM + Kimi-K3 的续写腿（leg2）。Kimi-K3 的渲染器不认 continue_final_message，续写不能走
/v1/chat/completions，改成四步：

1. /v1/chat/completions/render：用原始请求渲染出 prompt 的 token id（带着原请求的
   chat_template_kwargs、tools，prompt 末尾已经打开了思考通道）；
2. /tokenize：把已经发给客户端的 reasoning/content 拼成的前缀文本编成 token id，接在 prompt 后面；
3. /inference/v1/generate：纯 token 进、纯 token 出的流式生成，不做任何解析；
4. /v1/chat/completions/derender：把生成的 token id 还原成 OpenAI 格式的 chunk（reasoning /
   content / tool_calls 的拆分由服务端的 parser 完成）。

这条路径依赖 CONTINUATION_URL 指向的地址同时暴露上面四个端点（服务端需要 --enable-scale-out）。

几处和 chat-continue 路径不同的设计：

- derender 的聚合：parser 路径下每次 derender 调用都会新建 parser、重放此前全部输出 token，请求体
  还带着整个 prompt 的 token id，逐 chunk 调用的总开销随输出长度平方增长。所以每累计
  config.DERENDER_BATCH_CHUNKS 个 generate chunk 才调用一次，遇到带 finish_reason 的结束 chunk
  时把剩余内容全部送出。第二条腿本身不处理卡住/崩溃，这里也不设「多久没有新 chunk 就强制送出」。
- K3 标记的尾部顺延：generate 的 chunk 很小，K3 的控制符 `<|open|>NAME<|sep|>` /
  `<|close|>NAME<|sep|>` 可能被切在「控制符 + 名字」和 `<|sep|>` 之间；derender 解码时会在控制符和
  文本之间插空格，服务端 parser 对「标记只来了一半」的扣留是按精确前缀匹配的，带了空格就失配，
  半截标记会漏进 reasoning / content 而且撤不回来。所以每次送出前，若末尾是控制符 + 名字，就把这
  两个 token 留到下一次再送。只有送出的批尾有这个风险，批内部的标记在同一次解码里是完整的。
  聚合和顺延是同一个送出算法，见 plan_flush()。
- 出口拆分：derender 一批 token 可能同时跨 reasoning、content、tool_calls，parser 会把它们合并
  成一个 delta；而流式 /v1/chat/completions 的一个 chunk 一次只带一种内容，所以网关按
  reasoning → content → tool_calls 的顺序拆成多个 chunk 再写给客户端，见 split_delta_chunk()。
- usage：和 chat-continue 路径是同一套公式（usage.correct_usage()）：被救回部分（leg1）一律按字符估算，
  不为了算 token 数多调 /tokenize；差别只在 leg2 的 reasoning_tokens——chat-continue 路径直接信任续写腿
  上报的值，这条路径的 generate 和 derender 都不上报，所以改按 derender 输出的 reasoning 文本做字符估算。
  这条路径用到的 /tokenize 只有两类：每次续写把前缀文本编成 token id（接在 prompt 后面发给 generate，
  这是 render-generate 本身需要的），以及首次使用时取 K3 控制 token 的 id（按 URL 和模型缓存）。
"""

import asyncio
import json
import logging
from typing import Callable, Optional

from aiohttp import (
    ClientConnectionError,
    ClientError,
    ClientPayloadError,
    ClientSession,
    ServerDisconnectedError,
    web,
)

from . import config
from .reconstruct import XTML_CLOSE, XTML_OPEN, XTML_SEP, build_k3_render_prefix
from .sse import REASONING_FIELDS, LineSplitter, StreamState
from .usage import correct_usage, estimate_tokens, rewrite_chunk_fields

log = logging.getLogger("continuation_gateway")

_STREAM_ERRORS = (ClientPayloadError, ConnectionResetError, ServerDisconnectedError, ClientConnectionError)


class RenderGenerateError(Exception):
    """某一步失败。reason 直接作为 FAILED 日志的 reason 字段（leg2_connect_error、
    leg2_http_<code> 之类，和 chat-continue 路径的取值一致，下游的日志关联工具不用另外认），
    detail 是补充说明。"""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class ClientWriteError(Exception):
    """往客户端写 chunk 失败（客户端已经断开）。和 generate 流本身断开分开：前者不是下游的问题，
    原样往上抛，由调用方按「意料之外的异常」处理（和 chat-continue 路径写客户端失败时一致）。"""


class K3Tokens:
    """Kimi-K3 控制 token 的 id，用服务端 /tokenize 取：三个控制符各是一个 token，
    `<|open|>think<|sep|>` 和 `<|open|>response<|sep|>` 各是三个 token。任何一个不符合都说明
    tokenizer 和这里的假设对不上，直接报错，不悄悄不生效。"""

    __slots__ = ("open_id", "close_id", "sep_id", "think_open", "response_open")

    def __init__(self, open_id, close_id, sep_id, think_open, response_open):
        self.open_id = open_id
        self.close_id = close_id
        self.sep_id = sep_id
        self.think_open = think_open
        self.response_open = response_open

    @property
    def marker_ids(self) -> tuple:
        return (self.open_id, self.close_id, self.sep_id)


_k3_tokens_cache = {}


# ---- 纯函数：送出算法、出口拆分、token 计数 -------------------------------------------------


def marker_tail_len(token_ids: list, marker_ids: tuple) -> int:
    """token_ids 末尾需要顺延到下一次送出的 token 数：末尾是 <|open|>/<|close|> 加一个名字
    token（最后一个既不是 <|sep|> 也不是控制符）时是 2，否则 0。单独停在末尾的 <|open|> 或
    <|close|> 是精确前缀，服务端能扣住，不需要顺延。"""
    open_id, close_id, _ = marker_ids
    if (len(token_ids) >= 2 and token_ids[-2] in (open_id, close_id)
            and token_ids[-1] not in marker_ids):
        return 2
    return 0


def plan_flush(pending: list, buffered: list, finish: bool,
               hold_tail: Callable[[list], int]) -> tuple:
    """一次送出：把上次顺延留下的 pending 和本批累计的 token 合在一起，返回 (这次要送去
    derender 的 token, 新的 pending)。finish 为真（结束 chunk）时全部送出，不再顺延；否则由
    hold_tail 决定末尾留几个 token 到下一次。hold_tail 是模型相关的部分（K3 传 marker_tail_len
    的闭包），其余逻辑和模型无关。"""
    token_ids = pending + buffered
    if finish or not token_ids:
        return token_ids, []
    keep = hold_tail(token_ids)
    if keep <= 0:
        return token_ids, []
    return token_ids[:-keep], token_ids[-keep:]


def split_delta_chunk(chunk: dict) -> list:
    """把 derender 返回的一个 chunk 拆成按序的多个 chunk：流式 /v1/chat/completions 一个 chunk
    一次只带一种内容，而 derender 一批 token 跨多个部分时会合成一个 delta。顺序固定为
    reasoning → content → tool_calls（每个 tool_call 条目单独一个 chunk，条目内的 name 和
    arguments 不再拆）。role 留在拆出的第一个 chunk 上，finish_reason 和 usage 放在最后一个上，
    id、model 等其余字段每个都带。只有零个或一个部分的 chunk 原样返回。"""
    choices = chunk.get("choices") or []
    if len(choices) != 1:
        return [chunk]
    choice = choices[0]
    delta = choice.get("delta") or {}
    parts = []
    reasoning = {name: delta[name] for name in REASONING_FIELDS if delta.get(name)}
    if reasoning:
        parts.append(reasoning)
    if delta.get("content"):
        parts.append({"content": delta["content"]})
    for tool_call in delta.get("tool_calls") or []:
        parts.append({"tool_calls": [tool_call]})
    if len(parts) <= 1:
        return [chunk]

    known = {"role", "content", "tool_calls", *REASONING_FIELDS}
    extras = {k: v for k, v in delta.items() if k not in known and v is not None}
    head = {k: v for k, v in chunk.items() if k not in ("choices", "usage")}
    pieces = []
    for i, part in enumerate(parts):
        piece_delta = {}
        if i == 0:
            if delta.get("role"):
                piece_delta["role"] = delta["role"]
            piece_delta.update(extras)
        piece_delta.update(part)
        piece_choice = {"index": choice.get("index", 0), "delta": piece_delta, "finish_reason": None}
        if i == 0 and choice.get("logprobs") is not None:
            piece_choice["logprobs"] = choice["logprobs"]
        piece = dict(head)
        piece["choices"] = [piece_choice]
        if i == len(parts) - 1:
            piece_choice["finish_reason"] = choice.get("finish_reason")
            if chunk.get("usage"):
                piece["usage"] = chunk["usage"]
        pieces.append(piece)
    return pieces


def thinking_open_from_prompt(prompt_ids: list, k3: K3Tokens) -> bool:
    tail = prompt_ids[-3:]
    if tail == k3.think_open:
        return True
    if tail == k3.response_open:
        return False
    raise RenderGenerateError("leg2_render_prompt_unrecognized", f"prompt tail token ids {tail}")


# ---- 与服务端交互 -----------------------------------------------------------------------


def _json_headers(headers: dict) -> dict:
    """转发客户端原请求里的头（鉴权等），但这几个端点的请求/响应都是网关自己的 JSON：去掉客户端的
    Accept/Accept-Encoding/Content-Type，并要求不压缩（ClientSession 关了自动解压）。"""
    result = {k: v for k, v in headers.items()
              if k.lower() not in ("accept", "accept-encoding", "content-type")}
    result["Accept-Encoding"] = "identity"
    return result


async def _post_json(session: ClientSession, url: str, headers: dict, body: dict, step: str) -> dict:
    try:
        async with session.post(url, headers=headers, json=body, allow_redirects=False) as resp:
            if resp.status != 200:
                text = (await resp.text())[:300]
                raise RenderGenerateError(f"leg2_http_{resp.status}", f"step={step} body={text!r}")
            try:
                result = await resp.json(content_type=None)
            except ValueError as e:
                raise RenderGenerateError("leg2_render_bad_response", f"step={step}: {e}")
    except (ClientError, asyncio.TimeoutError) as e:
        raise RenderGenerateError("leg2_connect_error", f"step={step}: {e}")
    if not isinstance(result, dict):
        raise RenderGenerateError("leg2_render_bad_response", f"step={step}: not a JSON object")
    return result


async def _tokenize(session: ClientSession, base: str, headers: dict, model: str, text: str) -> list:
    out = await _post_json(session, base + "/tokenize", headers,
                           {"model": model, "prompt": text, "add_special_tokens": False}, "tokenize")
    tokens = out.get("tokens")
    if not isinstance(tokens, list):
        raise RenderGenerateError("leg2_render_bad_response", "step=tokenize: no 'tokens' list")
    return tokens


async def get_k3_tokens(session: ClientSession, base: str, headers: dict, model: str) -> K3Tokens:
    key = (base, model)
    cached = _k3_tokens_cache.get(key)
    if cached is not None:
        return cached
    markers = await _tokenize(session, base, headers, model, f"{XTML_OPEN}{XTML_CLOSE}{XTML_SEP}")
    if len(markers) != 3:
        raise RenderGenerateError("leg2_tokenizer_mismatch",
                                  f"'{XTML_OPEN}{XTML_CLOSE}{XTML_SEP}' tokenizes to {markers}, "
                                  f"expected 3 single-token control markers")
    open_id, close_id, sep_id = markers
    openers = await _tokenize(session, base, headers, model,
                              f"{XTML_OPEN}think{XTML_SEP}{XTML_OPEN}response{XTML_SEP}")
    if (len(openers) != 6 or openers[0] != open_id or openers[2] != sep_id
            or openers[3] != open_id or openers[5] != sep_id):
        raise RenderGenerateError("leg2_tokenizer_mismatch",
                                  f"'{XTML_OPEN}think{XTML_SEP}{XTML_OPEN}response{XTML_SEP}' tokenizes to "
                                  f"{openers}, expected [open, <think>, sep, open, <response>, sep]")
    k3 = K3Tokens(open_id, close_id, sep_id, openers[:3], openers[3:])
    _k3_tokens_cache[key] = k3
    return k3


class _DerenderRelay:
    """把 generate 的 token 按 plan_flush 的节奏送去 derender，再把结果拆分后写给客户端。"""

    def __init__(self, session: ClientSession, url: str, headers: dict, model: str, chat_request: dict,
                 prompt_ids: list, hold_tail: Callable[[list], int], out: web.StreamResponse,
                 response_id: Optional[str], reasoning_field: Optional[str]):
        self.session = session
        self.url = url
        self.headers = headers
        self.model = model
        self.chat_request = chat_request
        self.prompt_ids = prompt_ids
        self.hold_tail = hold_tail
        self.out = out
        self.response_id = response_id
        self.reasoning_field = reasoning_field
        self.pending = []
        self.buffered = []
        self.buffered_chunks = 0
        self.stream_state = None
        self.finished = False
        self.derender_calls = 0
        self.last_chunk = {}
        self.reasoning_text = []  # 写给客户端的 reasoning 文本，用来估算续写段的 reasoning_tokens

    def add(self, token_ids: list) -> None:
        self.buffered.extend(token_ids)
        self.buffered_chunks += 1

    @property
    def batch_full(self) -> bool:
        return self.buffered_chunks >= config.DERENDER_BATCH_CHUNKS

    async def flush(self, finish_reason: Optional[str]) -> None:
        to_send, self.pending = plan_flush(self.pending, self.buffered, finish_reason is not None,
                                           self.hold_tail)
        self.buffered = []
        self.buffered_chunks = 0
        if not to_send and finish_reason is None:
            return
        out = await _post_json(self.session, self.url, self.headers, {
            "stream": True,
            "model": self.model,
            "generate_chunk": {"choices": [{"index": 0, "token_ids": to_send,
                                            "finish_reason": finish_reason}]},
            "stream_state": self.stream_state,
            "prompt_tokens": len(self.prompt_ids),
            "prompt_token_ids": self.prompt_ids,
            "chat_request": self.chat_request,
        }, "derender")
        self.derender_calls += 1
        chunk = out.get("chunk")
        if not isinstance(chunk, dict) or "stream_state" not in out:
            raise RenderGenerateError("leg2_render_bad_response", "step=derender: no chunk/stream_state")
        self.stream_state = out["stream_state"]
        await self._emit(chunk)

    async def _emit(self, chunk: dict) -> None:
        for piece in split_delta_chunk(chunk):
            rewrite_chunk_fields(piece, self.response_id, self.reasoning_field)
            for choice in piece.get("choices") or []:
                delta = choice.get("delta") or {}
                for name in REASONING_FIELDS:
                    if delta.get(name):
                        self.reasoning_text.append(delta[name])
                        break
            await self.write_chunk(piece)
            for choice in piece.get("choices") or []:
                if choice.get("finish_reason"):
                    self.finished = True
        self.last_chunk = chunk

    async def write_chunk(self, obj: dict) -> None:
        try:
            await self.out.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
        except _STREAM_ERRORS as e:
            raise ClientWriteError(str(e)) from e


async def run_render_generate(session: ClientSession, headers: dict, original_payload: dict,
                              state: StreamState, out: web.StreamResponse, recovered,
                              remaining_budget: Optional[int], log_id: str = "") -> tuple:
    """发起并转发整条续写腿，返回 (outcome, finished)，含义和 server.relay_leg2() 一致：outcome
    是这条腿结束的原因（给日志用），finished 是结束前有没有送出过 finish_reason。某一步失败抛
    RenderGenerateError，由调用方记日志。recovered 是被救回部分的估算 token 数（usage 修正用），
    remaining_budget 是客户端预算扣掉被救回部分后剩下的（客户端没传预算时为 None），都由调用方
    和 chat-continue 路径用同一套逻辑算好。"""
    base = config.CONTINUATION_URL
    model = original_payload.get("model")
    json_headers = _json_headers(headers)
    chat_request = {k: v for k, v in original_payload.items() if k not in ("stream", "stream_options")}

    rendered = await _post_json(session, base + "/v1/chat/completions/render", json_headers,
                                chat_request, "render")
    prompt_head = rendered.get("token_ids")
    if not isinstance(prompt_head, list) or not prompt_head:
        raise RenderGenerateError("leg2_render_bad_response", "step=render: no token_ids")

    k3 = await get_k3_tokens(session, base, json_headers, model)
    thinking_open = thinking_open_from_prompt(prompt_head, k3)
    try:
        prefix = build_k3_render_prefix(state.reasoning, state.content, thinking_open)
    except ValueError as e:
        raise RenderGenerateError("leg2_render_prefix_mismatch", str(e))
    prefix_ids = await _tokenize(session, base, json_headers, model, prefix)
    prompt_ids = prompt_head + prefix_ids

    # 客户端传了预算：用调用方算好的剩余预算。没传时 render 可能已经按 max_model_len 减 prompt 长度
    # 填了一个上限，prompt 接上前缀变长了，上限也要同步减掉（这里的前缀 token 数只用来保证不超
    # 上下文长度，不参与计费）。
    sampling_params = dict(rendered.get("sampling_params") or {})
    if remaining_budget is not None:
        sampling_params["max_tokens"] = remaining_budget
    elif isinstance(sampling_params.get("max_tokens"), int):
        sampling_params["max_tokens"] = max(1, sampling_params["max_tokens"] - len(prefix_ids))

    log.info("[continuation req=%s] render-generate: thinking_open=%s prompt_tokens=%d prefix_tokens=%d "
             "max_tokens=%s", log_id, thinking_open, len(prompt_head), len(prefix_ids),
             sampling_params.get("max_tokens"))

    generate_body = dict(rendered)
    generate_body.pop("token_offsets", None)  # 按原 prompt 算的偏移，接上前缀后已不对应
    generate_body.update({
        "token_ids": prompt_ids,
        "model": model,
        "sampling_params": sampling_params,
        "stream": True,
        "stream_options": {"include_usage": True},
    })

    relay = _DerenderRelay(
        session, base + "/v1/chat/completions/derender", json_headers, model, chat_request, prompt_ids,
        lambda ids: marker_tail_len(ids, k3.marker_ids), out, state.response_id, state.reasoning_field)
    generated_ids = []
    generate_usage = None
    end_reason = "clean EOF"
    try:
        async with session.post(base + "/inference/v1/generate", headers=json_headers,
                                json=generate_body, allow_redirects=False) as resp:
            if resp.status != 200:
                text = (await resp.text())[:300]
                raise RenderGenerateError(f"leg2_http_{resp.status}", f"step=generate body={text!r}")
            splitter = LineSplitter()
            done = False
            async for raw in resp.content.iter_any():
                for line in splitter.feed(raw):
                    if not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:"):].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        raise RenderGenerateError("leg2_incomplete", f"step=generate error={obj['error']!r}")
                    if obj.get("usage"):
                        generate_usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        token_ids = choice.get("token_ids") or []
                        finish_reason = choice.get("finish_reason")
                        if not token_ids and not finish_reason:
                            continue
                        generated_ids.extend(token_ids)
                        relay.add(token_ids)
                        if finish_reason or relay.batch_full:
                            await relay.flush(finish_reason)
                if done:
                    end_reason = "[DONE]"
                    break
    except _STREAM_ERRORS as e:
        end_reason = f"generate stream disconnected ({type(e).__name__})"
    except (ClientError, asyncio.TimeoutError) as e:
        raise RenderGenerateError("leg2_connect_error", f"step=generate: {e}")

    log.info("[continuation req=%s] render-generate: generated_tokens=%d derender_calls=%d "
             "(batch=%d chunks) finished=%s", log_id, len(generated_ids), relay.derender_calls,
             config.DERENDER_BATCH_CHUNKS, relay.finished)
    if not relay.finished:
        return end_reason, False

    completion_tokens = (generate_usage or {}).get("completion_tokens")
    if not isinstance(completion_tokens, int):
        completion_tokens = len(generated_ids)
    final_leg_usage = {
        "prompt_tokens": (generate_usage or {}).get("prompt_tokens", len(prompt_ids)),
        "completion_tokens": completion_tokens,
        "completion_tokens_details": {"reasoning_tokens": estimate_tokens(
            "".join(relay.reasoning_text), config.CJK_CHARS_PER_TOKEN, config.OTHER_CHARS_PER_TOKEN)},
    }
    usage = correct_usage(recovered, final_leg_usage, False, "vllm")
    last = relay.last_chunk
    await relay.write_chunk({"id": state.response_id or last.get("id"), "object": "chat.completion.chunk",
                        "created": last.get("created"), "model": last.get("model", model),
                        "choices": [], "usage": usage})
    try:
        await out.write(b"data: [DONE]\n\n")
    except _STREAM_ERRORS as e:
        raise ClientWriteError(str(e)) from e
    return end_reason, True
