"""Kimi-K3 崩溃续写网关 —— v1 范围：只处理"崩溃时还没出现过 tool_call chunk、且请求本身没有
用 response_format 约束成 json_object/json_schema"的情况（普通自由文本的 content-done /
thinking-partial，多模态请求——带图片/音频/视频这类非文本内容——也在覆盖范围内）。两类场景
明确排除，都在 `should_intervene()`/`state.tool_calls_seen` 里挡掉：

- tool_call：`continue_final_message` 标准语义下 `tool_calls` 字段代表"这轮已经说完、该
  tool 角色回复了"，没有"还在生成中"的状态位，要支持得改 SGLang 源码，这里不做（这条崩溃时
  才能发现，出现过 tool_call chunk 就不再救）。
- response_format 是 json_object/json_schema 的结构化输出：不打算改 SGLang 支持（跟
  tool_call 不同，这条纯粹是不想做），而且被约束的 JSON 内容混在普通 content chunk 里流
  出来，跟自由文本没有独立信号能区分，网关这层要单独识别"这段 content 其实是被 guided
  decoding 约束着的"再决定怎么重建前缀，复杂度不值得。这条从原始请求的 response_format
  字段就能直接判断，不用等流式过程中才发现，在 `should_intervene()` 里前置排除。

多模态请求（messages 里带图片/音频/视频这类非文本 content-part）走跟纯文本请求相同的续写
编排，但 usage 修正换了一套算法：usage.py 的 estimate_tokens() 只按字符数估算 token，对
非文本内容没有对应的字符数可数，估算出来的 prompt_tokens 不可信（实测过真实 6 万多 token
的图片请求，估算结果只有 12）；`_has_multimodal_content()` 判断出的结果会一路传到
`usage.correct_usage()`，多模态请求改用"续写腿真实上报的 prompt_tokens 减去被救回内容的
估算 token 数"这种减法，而不是纯文本请求那套"估算值直接当 prompt_tokens"的算法，具体见
usage.py `correct_usage()` 顶部注释。这条判断跟大 body 排除（MAX_CONTINUATION_BODY_MB）
互不影响——多模态请求如果 body 超限，仍然会被那条规则挡在续写范畴外，两条规则各管各的。

跑法（下游预期是一个机房/集群路由网关，不是直连某个具体 SGLang 实例；原始请求和续写请求打
同一个 URL，实例选择/避开故障实例交给下游网关负责，这一层不自己维护 SGLang 实例列表）：

    DOWNSTREAM_URL=http://<路由网关>:<port> CONTINUATION_MODELS=<Kimi-K3 的 model 名字> \\
        python -m continuation_gateway.server

usage 修正 / max_tokens 扣减用到的 token 数不是靠 /v1/tokenize 现测的（下游不想为这个改
服务，且 messages 模式在 Kimi-K3 部署上一直有已知问题），是按字符数估算的，见 usage.py
顶部注释。
"""

import asyncio
import json
import logging
import sys
import time
import uuid

from aiohttp import (
    ClientConnectionError,
    ClientError,
    ClientPayloadError,
    ClientSession,
    ClientTimeout,
    ServerDisconnectedError,
    TCPConnector,
    web,
)

from . import config
from .reconstruct import build_prefix, classify_case, needs_thinking_disabled
from .sse import LineSplitter, StreamState, feed_line
from .usage import estimate_recovered_tokens, rewrite_leg2_line

if not config.DOWNSTREAM_URL:
    sys.exit("Set DOWNSTREAM_URL, e.g. DOWNSTREAM_URL=http://172.26.3.82:8050 "
             "python -m continuation_gateway.server")

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host",
    # content-length 也要剥掉：转发出去的 body 不一定跟客户端原始请求等长（续写腿的 body是
    # 网关自己重新拼的 JSON），aiohttp 会直接信任显式传入的 Content-Length 而不是按实际
    # body 重新计算（实测过），带着原始请求的旧值走会把续写请求的 body 截断/发坏。
    "content-length",
}

# 卡住(idle timeout)和断连，触发续写的判定方式不同，但后果一样：都用它们判断"这条腿是不是
# 没能正常说完就断了"，具体见 timed_reads/relay 的处理——不区分对待。
DISCONNECT_ERRORS = (ClientPayloadError, ConnectionResetError, ServerDisconnectedError, ClientConnectionError)

# 日志统一用北京时间（UTC+8，没有夏令时，固定偏移就够）——部署容器的系统时区通常是 UTC，
# 不改的话日志时间戳跟实际发生时间对不上，排查问题时得手动加 8 小时。
logging.Formatter.converter = lambda *args: time.gmtime(time.time() + 8 * 3600)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("continuation_gateway")


def filter_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def _has_multimodal_content(payload: dict) -> bool:
    """粗略判断这条请求是不是带了非文本内容（图片/音频/视频这类 content-part）。不用来排除
    续写覆盖——多模态请求走跟纯文本请求相同的续写编排——而是决定 usage 修正用哪套算法：
    usage.py 的 estimate_tokens() 只按字符数估算 token，`_message_text()` 里对非 text 的
    content-part 是直接跳过的，这些内容压根没有对应的字符数可数，估算出来的 prompt_tokens
    对多模态请求不可信（实测过一条带图片的真实请求，真实 prompt 有六万多 token，估算结果
    只有 12）。这个函数的结果会传给 `attempt_continuation()`/`usage.correct_usage()`，多模态
    请求改用"续写腿真实上报的 prompt_tokens 减去被救回内容的估算 token 数"这种减法，具体见
    usage.py `correct_usage()` 顶部注释。"""
    for message in payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") not in (None, "text"):
                return True
    return False


def should_intervene(payload: dict, body_size: int) -> bool:
    # 每个 return False 分支都要留一行日志——这个函数决定一条请求是走 guarded()（有续写
    # 覆盖，后续日志走"[continuation req=...]"那一套）还是 passthrough()（纯转发，
    # 只有 passthrough() 自己那一行日志），任何一个分支悄悄漏判都会导致事后查不到"这条
    # 请求当时为什么没进续写覆盖"。
    model = (payload.get("model") or "").lower()
    if not payload.get("stream"):
        log.info("model=%s non-stream request, skipping continuation coverage (passthrough only)",
                  payload.get("model"))
        return False
    if payload.get("continue_final_message"):
        # 客户端自己发的续写请求不再续写：最后一条 assistant 消息的 partial 内容是客户端
        # 自己拼的，格式不可控，跟这里的重建逻辑假设的"partial 状态来自一次正常、未续写过
        # 的流式响应"这个前提不符；而且 usage 修正公式假设"恰好两条腿"，如果在客户端自己的
        # 续写之上再续一次，没法可靠知道客户端那次续写已经消耗了多少 token，会破坏公式。
        log.info("model=%s client already sent continue_final_message, skipping continuation "
                  "coverage (passthrough only)", payload.get("model"))
        return False
    if not config.CONTINUATION_MODELS or model not in config.CONTINUATION_MODELS:
        log.info("model=%s not in CONTINUATION_MODELS=%s, skipping continuation coverage "
                  "(passthrough only)", payload.get("model"), config.CONTINUATION_MODELS or "<none configured>")
        return False
    if (payload.get("response_format") or {}).get("type") in ("json_object", "json_schema"):
        # 结构化输出（json_object/json_schema）不进续写，且不打算靠改 SGLang 支持——这条跟
        # tool_call 的排除理由不一样：tool_call 至少理论上能通过改 SGLang 源码支持，这条纯粹
        # 是不想做。而且约束到的 JSON 内容是混在普通 content chunk 里流出来的，跟自由文本长得
        # 一样，没有独立的字段/信号能分开抓，要在网关这一层单独识别"这段 content 其实是被
        # guided decoding 约束着的 JSON 片段"再决定怎么重建前缀，复杂度不值得。这个判断从
        # 原始请求的 response_format 字段就能直接拿到，不用等流式过程中才发现（不像 tool_call
        # 要等实际出现过 tool_call chunk 才知道），所以放在这里跟其它前置排除条件一起判断。
        log.info("response_format.type=%s, skipping continuation coverage for this request "
                  "(passthrough only): model=%s", payload["response_format"].get("type"),
                  payload.get("model"))
        return False
    if body_size > config.MAX_CONTINUATION_BODY_BYTES:
        # 大 body（典型是内嵌图片/视频的多模态请求）不进续写：解析后的 payload dict 要在
        # 内存里拿着直到流结束，续写真发生时 messages 数组还要原样再序列化/传输两次，body
        # 越大这个代价越不划算。
        log.info("request body %d bytes exceeds MAX_CONTINUATION_BODY_MB=%d, skipping "
                  "continuation coverage for this request (passthrough only): model=%s",
                  body_size, config.MAX_CONTINUATION_BODY_MB, payload.get("model"))
        return False
    return True


async def timed_reads(content, idle_timeout: float, stall_reason: list):
    """收到过至少一个 chunk 之后才会被用到（见 relay()/relay_leg2() 里第一个 chunk 的单独
    处理）：用 idle_timeout 逐块 yield 原始字节，超时/断连时直接 return（不抛异常）。调用方
    靠"有没有见过 finish_reason"判断是否需要续写，不需要区分具体是超时、断连、还是干净 EOF
    导致的循环结束——这三种情况的后续处理完全一样，只是把具体原因写进 stall_reason（长度
    0 或 1 的 list，用来在生成器提前 return 时把"为什么停"带回调用方打日志用——async
    generator 没法像普通 generator 那样通过 StopIteration.value 带返回值）。
    """
    while True:
        try:
            chunk = await asyncio.wait_for(content.readany(), timeout=idle_timeout)
        except asyncio.TimeoutError:
            stall_reason.append(f"idle timeout after {idle_timeout}s")
            return
        except DISCONNECT_ERRORS as e:
            stall_reason.append(f"downstream disconnected ({type(e).__name__})")
            return
        if not chunk:
            stall_reason.append("clean EOF")
            return
        yield chunk


async def _stream_downstream_verbatim(request: web.Request, downstream) -> web.StreamResponse:
    resp_headers = filter_headers(downstream.headers)
    resp_headers.pop("Content-Length", None)
    out = web.StreamResponse(status=downstream.status, headers=resp_headers)
    await out.prepare(request)
    try:
        async for chunk in downstream.content.iter_any():
            await out.write(chunk)
        await out.write_eof()
    except (*DISCONNECT_ERRORS, asyncio.CancelledError):
        downstream.close()
        raise
    finally:
        downstream.release()
    return out


async def passthrough(request: web.Request, raw_body: bytes = None) -> web.StreamResponse:
    session: ClientSession = request.app["session"]
    target = config.DOWNSTREAM_URL + request.path_qs
    # 这一层不解析 body（catch_all 打过来的请求甚至不一定是 JSON），所以日志只能落到
    # method/path 这个粒度——但这条请求"走的是纯转发、没有续写覆盖"这件事本身必须留痕，
    # 不然配合 should_intervene() 那些 skip 日志也拼不出这条请求的完整去向。
    log.info("PASSTHROUGH %s %s -> %s", request.method, request.path_qs, target)
    headers = filter_headers(request.headers)
    data = raw_body if raw_body is not None else (request.content if request.can_read_body else None)
    del raw_body  # 用完就扔，见 chat_completions() 里同样处理的注释
    try:
        downstream = await session.request(request.method, target, headers=headers, data=data,
                                          allow_redirects=False)
    except (ClientError, asyncio.TimeoutError) as e:
        log.warning("PASSTHROUGH %s %s failed: %s", request.method, request.path_qs, e)
        # 429 而不是 502：理由跟 guarded() 里 leg1_connect_error 那处一样（见那边的完整
        # 注释）——前面的 nginx 只在 5xx 上做绕开两层网关直连 SGLang 的重试，429 能避开
        # 这条规则，把"这个集群现在连不上下游"如实反映给最上层调用方去做 Provider 级别的
        # 失败转移，而不是在已经吃紧的下游上再空转两轮重试。
        return web.Response(status=429, text=f"downstream error: {e}\n")
    del data
    return await _stream_downstream_verbatim(request, downstream)


async def relay(downstream_content, out: web.StreamResponse, state: StreamState,
                 splitter: LineSplitter, req_id: str) -> tuple:
    """原始腿：先无限等第一个 chunk——这一层不对"迟迟没有第一个 chunk"这件事负责，等多久、
    要不要重试是上层的事；这期间发生的连接异常也不吞，直接往上抛，让这次请求按普通失败处理，
    不进入续写逻辑（客户端还没看到任何内容，没什么好"救"的）。收到第一个 chunk 之后才开始
    用 idle timeout 盯"卡住"这件事，字节原样转发给客户端（不改写任何内容），同一份字节喂
    SSE 行解析更新 state。

    idle timeout 本身不代表"该结束这条腿"——它只在"接下来有机会做一次有意义的续写"时才是
    一个值得停下来的信号，也就是要同时满足：还没见过 tool_call、且已经攒到了可恢复的
    reasoning/content。不满足这两条时（tool_call 已经出现，或者目前为止还什么实质内容都
    没有），idle timeout 只是又白等了一轮，继续等——跟没有这层网关时下游只是慢一样，不应该
    被这一层主动挂断，等下去仍然有机会等到内容。真正的断连/干净 EOF 不管当前是什么状态都会
    结束这个循环，因为已经没有字节可读了，继续等没有意义，这个终点和没有网关时一致（普通
    转发这时候也会结束）。

    返回 (needs_retry, stall_reason)：needs_retry 为 True 表示这条腿结束时已经转发过至少
    一个 chunk、但还没见过 finish_reason，调用方据此决定要不要续写；第一个 chunk 就干净
    EOF（downstream 一个字节都没吐）时 needs_retry 为 False，不算需要续写，此时 stall_reason
    是 None。`req_id` 用于每次 idle timeout 触发时打日志——不管这次超时最终是"继续等"还是
    "交给上层决定要不要续写"，都统一在这里留一行标记，不是只有前者才值得记；这时候
    `state.response_id` 大概率已经从第一个 chunk 里解析出来了，优先用它（跟客户端能看到的
    completion id 对上号），解析不出来才退回用这个兜底 id。
    """
    chunk = await downstream_content.readany()
    if not chunk:
        return False, None
    await out.write(chunk)
    for line in splitter.feed(chunk):
        feed_line(line, state)

    while True:
        stall_reason = []
        async for chunk in timed_reads(downstream_content, config.STALL_IDLE_TIMEOUT_SECONDS, stall_reason):
            await out.write(chunk)
            for line in splitter.feed(chunk):
                feed_line(line, state)
        reason = stall_reason[0]
        has_recoverable = not state.tool_calls_seen and (state.reasoning or state.content)
        if reason.startswith("idle timeout"):
            log_id = state.response_id or req_id
            if has_recoverable:
                # 这次超时是"可以做点什么"的那种——留一行标记这个事件本身发生过，具体决定
                # 续不续写由 guarded() 的 TRIGGERED 日志接着记，这里不重复那份细节。
                log.info("[continuation req=%s] leg1 idle timeout (%s), handing off for "
                         "continuation decision", log_id, reason)
            else:
                log.warning("[continuation req=%s] leg1 stalled (%s) but not actionable "
                            "(tool_calls_seen=%s, has_content=%s), continuing to wait",
                            log_id, reason, state.tool_calls_seen,
                            bool(state.reasoning or state.content))
                continue
        return state.finish_reason is None, reason


async def relay_leg2(downstream_content, out: web.StreamResponse, splitter: LineSplitter, recovered,
                      response_id: str, log_id: str, is_multimodal: bool = False) -> tuple:
    """续写腿：逐行转发（不是整块字节透传），因为带 usage 的那条 data 行、以及每一行的 id
    都要原地改写（id 改写见 usage.py:rewrite_leg2_line 顶部注释——续写腿是网关自己发起的
    新请求，下游会分配一个新 completion id，不改写的话客户端会看到 id 中途变化）。同样先
    无限等第一个 chunk 再开始用 idle timeout。is_multimodal 原样透传给 rewrite_leg2_line()，
    决定 usage 修正的 prompt_tokens 走估算值还是从真实 leg2 prompt_tokens 减算。

    单次续写策略下这条腿没有第三条腿可退，所以 idle timeout 在这里不该再有"结束这条腿"的
    效果——停下来也换不来任何补救动作，唯一自洽的选择是跟没有这层网关时一样继续等（下游
    真的只是慢的话，等下去仍然有机会把流正常说完）；每次 idle timeout 只打一行 warning 留痕
    （方便观测下游到底卡了多久、卡了几次），不代表放弃。真正的断连/干净 EOF 才会让这条腿
    结束，因为已经没有字节可读了，等也没用，这个终点和没有网关时一致。

    返回 (outcome, finished)：outcome 是这条腿结束的原因，纯粹给调用方打日志用；finished
    是有没有在结束前见过 finish_reason——单靠 outcome 的文字（比如"clean EOF"）分不清
    "正常说完后连接关闭"和"没说完就断了"，这里额外喂一份 StreamState 只为了拿这个信号。
    """
    state = StreamState()
    try:
        chunk = await downstream_content.readany()
    except DISCONNECT_ERRORS as e:
        return f"downstream disconnected before first chunk ({type(e).__name__})", False
    if not chunk:
        return "clean EOF before first chunk", False
    for line in splitter.feed(chunk):
        feed_line(line, state)
        await out.write(rewrite_leg2_line(line, recovered, response_id, is_multimodal) + b"\n")

    while True:
        stall_reason = []
        async for chunk in timed_reads(downstream_content, config.STALL_IDLE_TIMEOUT_SECONDS, stall_reason):
            for line in splitter.feed(chunk):
                feed_line(line, state)
                await out.write(rewrite_leg2_line(line, recovered, response_id, is_multimodal) + b"\n")
        reason = stall_reason[0]
        if reason.startswith("idle timeout"):
            log.warning("[continuation req=%s] leg2 stalled (%s), no third leg, continuing to wait",
                        log_id, reason)
            continue
        return reason, state.finish_reason is not None


async def attempt_continuation(session: ClientSession, target: str, headers: dict, original_payload: dict,
                                state: StreamState, out: web.StreamResponse, log_id: str) -> None:
    recovered = estimate_recovered_tokens(original_payload, state.reasoning, state.content,
                                           config.CJK_CHARS_PER_TOKEN, config.OTHER_CHARS_PER_TOKEN)
    # 多模态请求的 recovered.prompt_tokens 是按字符数估算的，对非文本 content-part 不可信，
    # usage.correct_usage() 遇到 is_multimodal=True 时会换成从 leg2 真实 prompt_tokens 减算，
    # 不使用这个估算值——但这里仍然照常算出来、照常打进日志，方便跟减算结果对照排查。
    is_multimodal = _has_multimodal_content(original_payload)
    # case（见 reconstruct.classify_case()）到这一步已经是确定真的会发第二条腿之后的状态了
    # （tool_call_seen/no_recoverable_content 两个排除分支已经在 guarded() 里过滤掉，v1
    # 范围内 state.tool_calls_seen 到这里必然是 False，传它只是让这次调用跟 classify_case()
    # 的完整签名保持一致——以后 v2 如果放开 tool_call 场景也能真的走到这个函数，这里不用改），
    # 目前实际只会落在 content-done/thinking-partial 二选一，直接决定了下面 build_prefix()
    # 走哪个分支、needs_thinking_disabled() 是不是为真——跟 TRIGGERED 那行的 case 字段用的
    # 是同一份判断依据，两处应该总是一致的（除非 state.content 在两次调用之间被改过，不应该
    # 发生）。
    case = classify_case(state.tool_calls_seen, state.reasoning, state.content)
    log.info("[continuation req=%s] ATTEMPT case=%s prompt=%d reasoning=%d content=%d "
              "total_recovered=%d multimodal=%s",
              log_id, case, recovered.prompt_tokens, recovered.reasoning_tokens,
              recovered.content_tokens, recovered.total_recovered, is_multimodal)

    # 客户端原始请求用的是 max_tokens 还是 max_completion_tokens（新旧两个字段，语义等价），
    # 续写请求就沿用同一个字段名去扣减，不额外发明一个默认预算：客户端两个都没传，意思就是
    # "不限制"，续写腿也不该无中生有地加一个上限。正常客户端只会传其中一个；万一两个都传了
    # （不常见），两个都按扣减后的值改，避免下游到底读哪个字段产生歧义。
    budget_fields = [f for f in ("max_tokens", "max_completion_tokens") if original_payload.get(f) is not None]

    prefix = build_prefix(original_payload.get("model"), state.reasoning, state.content)
    continuation_payload = dict(original_payload)
    continuation_payload["messages"] = list(original_payload.get("messages", [])) + [
        {"role": "assistant", "content": prefix}
    ]

    if budget_fields:
        original_budget = original_payload[budget_fields[0]]
        remaining_budget = original_budget - recovered.total_recovered
        if remaining_budget <= 0:
            log.info("[continuation req=%s] SKIPPED reason=budget_exhausted original=%d recovered=%d",
                      log_id, original_budget, recovered.total_recovered)
            return
        log.info("[continuation req=%s] budget original=%d remaining=%d", log_id, original_budget, remaining_budget)
        for field in budget_fields:
            continuation_payload[field] = remaining_budget
    # else: 客户端原始请求没有限制 token 数，续写请求也不设——不无中生有一个默认预算。

    continuation_payload["continue_final_message"] = True
    continuation_payload["add_generation_prompt"] = False
    continuation_payload["stream"] = True
    continuation_payload["stream_options"] = {"include_usage": True}
    if needs_thinking_disabled(state.content):
        continuation_payload["chat_template_kwargs"] = {"thinking": False}

    try:
        downstream = await session.post(target, headers=headers, json=continuation_payload,
                                       allow_redirects=False)
    except (ClientError, asyncio.TimeoutError):
        log.exception("[continuation req=%s] FAILED reason=leg2_connect_error case=%s", log_id, case)
        return

    if downstream.status != 200:
        log.warning("[continuation req=%s] FAILED reason=leg2_http_%s case=%s", log_id, downstream.status, case)
        downstream.release()
        return

    splitter = LineSplitter()
    try:
        leg2_outcome, leg2_finished = await relay_leg2(downstream.content, out, splitter, recovered,
                                                         state.response_id, log_id, is_multimodal)
    finally:
        downstream.release()
    if leg2_finished:
        log.info("[continuation req=%s] SUCCEEDED case=%s leg2_outcome=%s", log_id, case, leg2_outcome)
    else:
        log.warning("[continuation req=%s] FAILED reason=leg2_incomplete case=%s leg2_outcome=%s", log_id, case, leg2_outcome)


async def chat_completions(request: web.Request) -> web.StreamResponse:
    raw_body = await request.read()
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        payload = None

    # 两条分支都不能直接 `return await xxx(request, ..., raw_body)`：那样 chat_completions
    # 自己这个协程帧会一直挂在 await 上，它这份 raw_body 引用会跟被调用方内部那份一起活到
    # 整条流结束，被调用方内部再怎么 del 都没用（两份引用，少一份计数不会归零，实测过）。
    # 这里先创建 coroutine（参数已经绑进它自己的帧了）、扔掉这一层的引用，再去 await，才能让
    # passthrough()/guarded() 里的 del 真正在用完 raw_body 之后就把它回收掉。
    if payload is None or not should_intervene(payload, len(raw_body)):
        coro = passthrough(request, raw_body)
        del raw_body
        return await coro

    coro = guarded(request, payload, raw_body)
    del raw_body
    return await coro


async def guarded(request: web.Request, payload: dict, raw_body: bytes) -> web.StreamResponse:
    # 兜底 id，只在异常到"下游给的 completion id 拿不到"这种理论上不该发生的情况时才会被
    # 用到（正常情况下 needs_retry 分支走到时 state.response_id 早已经从 leg1 第一个 chunk
    # 里填好了，见下面 log_id 的取法）。不直接用这个当主 id 的原因：客户端收到的每个 chunk
    # 本来就带一个 "id" 字段（下游 SGLang 分配的 completion id），排查问题时最自然的锚点是
    # 客户端那边也看得到的这个值，而不是网关另外发明、只存在于日志里的标识符。
    req_id = uuid.uuid4().hex[:8]
    session: ClientSession = request.app["session"]
    target = config.DOWNSTREAM_URL + request.path_qs
    headers = filter_headers(request.headers)

    try:
        downstream = await session.post(target, headers=headers, data=raw_body, allow_redirects=False)
    except asyncio.CancelledError:
        # 客户端在网关还没连上下游之前就断开了（见 main() 里 handler_cancellation 的注释）
        # ——不是下游连接失败，是这个请求本身已经没人要了，不该算 leg1_connect_error（那个
        # reason 是留给"下游真的连不上/连太慢"这种下游侧问题的，混进去会误导以后靠这行日志
        # 排查下游健康状况）。必须原样 `raise`，不能吞掉——吞掉 CancelledError 会破坏任务
        # 取消的语义，让这次 cancel 看起来像是"处理完了"。
        log.info("[continuation req=%s] upstream disconnected while still connecting to "
                  "downstream, aborting connect attempt (not a downstream failure)", req_id)
        raise
    except (ClientError, asyncio.TimeoutError) as e:
        log.warning("[continuation req=%s] FAILED reason=leg1_connect_error: %s", req_id, e)
        # 这里连不上下游，返回 429 而不是 502
        # 部署环境里这个网关前面还有一层 nginx，规则是"收到 5xx 就重试，最多两次，重试目标
        # 直接打 SGLang pod、绕开这里和另一层网关"——用 502/503/504 这类 5xx 等于在下游已经
        # 扛不住新连接（这条异常本身就是这个信号）的时候，教 nginx 绕开两层网关的保护（也
        # 包括这里的续写覆盖）再多打两发直连请求，雪上加霜；两次重试大概率还是失败，兜一圈
        # 只是白白多占用下游的连接资源、多等一轮 CONNECT_TIMEOUT_SECONDS。429 是 4xx，nginx
        # 这条重试规则不会命中，失败会直接原样传回最上层调用方——业务侧确认过收到 429 会
        # 切到备用 API Provider，这正是"这个集群现在真扛不住，找别家"该有的信号，比在自己
        # 集群内部空转重试更合适。`leg1_http_<code>`（下游明确返回的非 200 状态码）不受
        # 这条影响，那条路径走的是 `_stream_downstream_verbatim()` 原样透传下游真实状态码，
        # 不是这里讨论的"网关自己决定发什么"。
        return web.Response(status=429, text=f"downstream error: {e}\n")
    # 原始字节只在上面这次 POST 里用一次——续写腿是从解析好的 payload dict 重建的，不需要
    # raw_body。并发多、body 又可能带 base64 图片这种大 payload 时，不早点扔掉这份引用的话，
    # 它会跟着这个协程一直活到整条流结束（可能是几十秒后），白占内存。
    del raw_body

    if downstream.status != 200:
        log.warning("[continuation req=%s] leg1 downstream returned status=%s before any streaming, "
                    "forwarding error response verbatim (no continuation attempted)",
                    req_id, downstream.status)
        return await _stream_downstream_verbatim(request, downstream)

    resp_headers = filter_headers(downstream.headers)
    resp_headers.pop("Content-Length", None)
    out = web.StreamResponse(status=downstream.status, headers=resp_headers)
    await out.prepare(request)

    state = StreamState()
    splitter = LineSplitter()
    try:
        needs_retry, stall_reason = await relay(downstream.content, out, state, splitter, req_id)
    except (*DISCONNECT_ERRORS, asyncio.CancelledError) as e:
        # 这里同时兜两类情况，都不属于续写范畴、都不吞异常：(1) 下游连接出问题——第一个
        # chunk 都没等到、或者读到一半断了（relay() 内部读下游失败，DISCONNECT_ERRORS）；
        # (2) 客户端（上游）断连——relay() 里往 out 写的时候失败也是同一批 DISCONNECT_ERRORS
        # 类型，不特意区分；如果客户端断连发生在我们正无限等第一个 chunk、还没开始往 out 写
        # 任何东西的时候，靠的是 asyncio.CancelledError——**这条路径要求 main() 里
        # `web.run_app(..., handler_cancellation=True)` 显式打开**（aiohttp 默认关闭，
        # 见那边的注释），不是"aiohttp 自带、不用配置就能用"的能力；这个参数不打开的话，
        # 客户端断连发生在等第一个 chunk 阶段时完全没反应，会一直等到自己配的超时（比如
        # CONNECT_TIMEOUT_SECONDS）才结束。两种情况处理方式一样：让这次请求按
        # 普通失败处理（跟没有这层网关时的行为一致），是否重试交给上层。downstream.close()
        # 确保这个已经坏掉的连接不会被当成好的放回连接池；下面的 finally 还会再调一次
        # release()，在已经 close() 过的连接上是安全的空操作。
        log_id = state.response_id or req_id
        log.info("[continuation req=%s] leg1 aborted (%s: %s), not attempting continuation "
                  "for this leg", log_id, type(e).__name__, e)
        downstream.close()
        raise
    finally:
        downstream.release()

    if needs_retry:
        # log_id 优先用 leg1 实际返回的 completion id（客户端在自己收到的每个 chunk 里也能
        # 看到同一个值，是网关日志和客户端抓包天然共享的锚点）；只有 leg1 第一个 chunk 就
        # 没能解析出 "id" 字段这种理论上不该出现的情况，才退回用上面生成的 req_id，保证这里
        # 无论如何都有一个非空值可以打日志、不会打出 "req=None"。
        log_id = state.response_id or req_id
        # TRIGGERED 只表示"leg1 没说完就断了"这个检测结果本身，不代表一定会真的发第二条腿——
        # 下面几个分支（tool_call/无内容）还会再排除掉一部分。这一行是排查"续写到底有没有
        # 发生、发生了几次"这类问题时唯一的入口：先 grep TRIGGERED 数一次演练触发了多少次，
        # 再挑感兴趣的 log_id（= 客户端看到的 completion id）grep 出这一个请求的完整生命
        # 周期日志，跟客户端自己记录的这个 id 对上号。case 字段（tool-call-seen/
        # content-done/thinking-partial/empty，见 reconstruct.classify_case()）是
        # "哪种情况的续写"这个问题最早能拿到答案的地方——即使后面被 tool_call_seen/
        # no_recoverable_content 排除掉、根本没有真的发第二条腿，这里仍然如实记录当时的
        # 状态，不因为最终没续写就不打。
        case = classify_case(state.tool_calls_seen, state.reasoning, state.content)
        log.info("[continuation req=%s] TRIGGERED reason=%s case=%s reasoning_chars=%d "
                  "content_chars=%d tool_calls_seen=%s model=%s",
                  log_id, stall_reason, case, len(state.reasoning),
                  len(state.content), state.tool_calls_seen, payload.get("model"))
        if state.tool_calls_seen:
            # v1 范围排除：已经出现过 tool_call chunk，不在网关能安全处理的范围内，不救。
            log.warning("[continuation req=%s] SKIPPED reason=tool_call_seen", log_id)
        elif state.reasoning or state.content:
            try:
                await attempt_continuation(session, target, headers, payload, state, out, log_id)
            except Exception:
                # 这是网关层，这里再崩会把已经转发给客户端的部分也带没了、甚至可能影响到
                # 同一个 worker 上别的请求；续写本身是"锦上添花"，救不回来不该拖累已经稳妥
                # 转发出去的正文。这里兜的是"意料之外"的错误（比如构造续写 payload 时的
                # 意外类型错误、客户端在续写腿写入过程中断开连接）。不用担心吞掉
                # asyncio.CancelledError——Python 3.8+ 它是 BaseException 的子类，不会被
                # `except Exception` 捕获，正常取消/关闭流程不受影响。
                log.exception("[continuation req=%s] FAILED reason=unexpected_exception case=%s",
                              log_id, case)
        else:
            # 收到过 chunk 但没有任何可用的 reasoning/content（比如只有一个空白的角色声明
            # chunk），没有实际内容可救，不属于续写范畴，流按原样收尾。
            log.info("[continuation req=%s] SKIPPED reason=no_recoverable_content", log_id)
    else:
        # 这条请求被判定为需要续写覆盖（should_intervene() 通过），但 leg1 从头到尾没有
        # 出问题：要么第一个 chunk 就是干净 EOF（stall_reason 为 None，下游一个字节都没吐，
        # 理论上不该发生但也不该静默）、要么正常见到了 finish_reason 后连接关闭。两种情况
        # 都没有触发 TRIGGERED，之前完全没有日志——不留痕的话，这类"啥事没有"的多数请求
        # 在日志里会跟"网关根本没处理过这条请求"没法区分。
        log_id = state.response_id or req_id
        log.info("[continuation req=%s] COMPLETED normally, no continuation needed (leg1_end=%s)",
                  log_id, stall_reason or "clean EOF on first read")

    try:
        await out.write_eof()
    except Exception:
        # 到这一步该做的都做完了，客户端这时候如果已经断开，收尾失败不算真正的错误，
        # 记一下就行，不用再往上抛。
        log.exception("failed to cleanly close the client stream (client likely gone): %s", target)
    return out


async def health(request: web.Request) -> web.Response:
    """网关自己的存活探测——纯 liveness，不碰下游。

    在 catch_all 之前单独注册这条路由，是为了把"网关进程本身活着"和"下游 SGLang 集群健不
    健康"这两件事分开：如果 /health 也像其它未知路径一样落进 catch_all 透传给下游，探测到的
    其实是下游状态——下游抖动/变慢时会把一个完全正常的网关进程也判定成不健康（被编排系统
    误重启），下游health接口异常时也无法反映网关自身是否存在问题（比如事件循环被某个请求
    卡住）却因为下游可达而看起来"健康"。能进到这个 handler 里跑起来并返回，本身就已经证明
    事件循环没有被卡死——不需要再额外做什么检查。
    """
    return web.json_response({"status": "ok"})


async def catch_all(request: web.Request) -> web.StreamResponse:
    return await passthrough(request)


async def on_startup(app: web.Application):
    app["session"] = ClientSession(
        auto_decompress=False,
        connector=TCPConnector(limit=config.DOWNSTREAM_CONNECTION_LIMIT),
        timeout=ClientTimeout(total=None, connect=config.CONNECT_TIMEOUT_SECONDS),
    )


async def on_cleanup(app: web.Application):
    await app["session"].close()


def main():
    app = web.Application(client_max_size=config.MAX_REQUEST_BODY_BYTES)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/chat/completions", chat_completions)
    # /health 必须在 catch_all 的通配路由之前注册，否则会被 catch_all 透传到下游。
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{tail:.*}", catch_all)
    log.info("listening on 0.0.0.0:%d, forwarding to %s (continuation models: %s)",
              config.PORT, config.DOWNSTREAM_URL, config.CONTINUATION_MODELS or "<none configured>")
    # handler_cancellation=True：aiohttp 的默认值是 False，默认值下客户端断连时
    # RequestHandler.connection_lost() 不会 cancel 正在处理这个请求的 task（读
    # aiohttp/web_protocol.py 源码确认，`_task_handler.cancel()` 那行套在
    # `if handler_cancellation and ...` 里）——不打开的话，网关卡在等 leg1 connect（比如
    # 下游一度没有健康实例、TCP connect 本身悬着不回）这种阶段时，客户端就算早就按自己的
    # 超时断开了，网关这边的 task 也完全没反应，只能一直等到自己配的 CONNECT_TIMEOUT_
    # SECONDS 才结束，白白多占几分钟的连接和请求状态。显式传 True 之后，客户端断连后不管
    # 这个 task 当时卡在哪个 await 上（等 leg1 connect、等 leg1 数据、等 leg2 数据……）都会
    # 被尽快 cancel，不用再靠自己配的各种超时兜底。**这个参数也要跟测试环境保持一致**——
    # `aiohttp.test_utils.TestServer` 内部默认就是 `handler_cancellation=True`，如果这里
    # 不显式设置，本地冒烟测试（用真实 `web.AppRunner`，不是 `test_utils`）跟生产环境用的
    # 会是两个不同的默认值，测试"验证过"的行为不能代表生产的真实行为。
    web.run_app(app, host="0.0.0.0", port=config.PORT, print=None, handler_cancellation=True)


if __name__ == "__main__":
    main()
