"""Kimi-K3 崩溃续写网关 —— v1 范围：只处理"崩溃时还没出现过 tool_call chunk、且请求本身没有
用 response_format 约束成 json_object/json_schema"的情况（普通自由文本的 content-done /
thinking-partial）。两类场景明确排除，都在 `should_intervene()`/`state.tool_calls_seen`
里挡掉：

- tool_call：`continue_final_message` 标准语义下 `tool_calls` 字段代表"这轮已经说完、该
  tool 角色回复了"，没有"还在生成中"的状态位，要支持得改 SGLang 源码，这里不做（这条崩溃时
  才能发现，出现过 tool_call chunk 就不再救）。
- response_format 是 json_object/json_schema 的结构化输出：不打算改 SGLang 支持（跟
  tool_call 不同，这条纯粹是不想做），而且被约束的 JSON 内容混在普通 content chunk 里流
  出来，跟自由文本没有独立信号能区分，网关这层要单独识别"这段 content 其实是被 guided
  decoding 约束着的"再决定怎么重建前缀，复杂度不值得。这条从原始请求的 response_format
  字段就能直接判断，不用等流式过程中才发现，在 `should_intervene()` 里前置排除。

不修改 proxy.py——这是独立的小项目，proxy.py 仍然是同事给的纯透明代理参考实现。

跑法（下游预期是一个机房/集群路由网关，不是直连某个具体 SGLang 实例；原始请求和续写请求打
同一个 URL，实例选择/避开故障实例交给下游网关负责，这一层不自己维护 SGLang 实例列表）：

    DOWNSTREAM_URL=http://<路由网关>:<port> CONTINUATION_MODELS=<Kimi-K3 的 model 名字> \\
        python -m continuation_gateway.server

还没法接实机验证，原因两条：
    1) 测试集群还没准备好；
    2) /v1/tokenize 的 messages 模式在 Kimi-K3 部署上目前返回 500（疑似 SGLang bug，待确认
       traceback）——这条链路打通前，usage 修正和 max_tokens 精确扣减会自动降级（见
       usage.py 顶部注释），不会因此打断给客户端的正文内容恢复。
"""

import asyncio
import json
import logging
import sys

from aiohttp import (
    ClientConnectionError,
    ClientError,
    ClientPayloadError,
    ClientSession,
    ClientTimeout,
    ServerDisconnectedError,
    web,
)

from . import config
from .reconstruct import build_prefix, needs_thinking_disabled
from .sse import LineSplitter, StreamState, feed_line
from .usage import measure_recovered_tokens, rewrite_usage_line

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("continuation_gateway")


def filter_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def should_intervene(payload: dict, body_size: int) -> bool:
    if not payload.get("stream"):
        return False
    if payload.get("continue_final_message"):
        # 客户端自己发的续写请求不再续写：最后一条 assistant 消息的 partial 内容是客户端
        # 自己拼的，格式不可控，跟这里的重建逻辑假设的"partial 状态来自一次正常、未续写过
        # 的流式响应"这个前提不符；而且 usage 修正公式假设"恰好两条腿"，如果在客户端自己的
        # 续写之上再续一次，没法可靠知道客户端那次续写已经消耗了多少 token，会破坏公式。
        return False
    if not config.CONTINUATION_MODELS or payload.get("model") not in config.CONTINUATION_MODELS:
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
        # 越大这个代价越不划算。只在"其他条件都满足、纯粹因为超限被排除"时才打这行日志，
        # 不会给占大多数的非续写流量（别的 model/非 stream 请求）添噪音。
        log.info("request body %d bytes exceeds MAX_CONTINUATION_BODY_MB=%d, skipping "
                  "continuation coverage for this request (passthrough only): model=%s",
                  body_size, config.MAX_CONTINUATION_BODY_MB, payload.get("model"))
        return False
    return True


async def timed_reads(content, idle_timeout: float):
    """收到过至少一个 chunk 之后才会被用到（见 relay()/relay_leg2() 里第一个 chunk 的单独
    处理）：用 idle_timeout 逐块 yield 原始字节，超时/断连时直接 return（不抛异常）。调用方
    靠"有没有见过 finish_reason"判断是否需要续写，不需要区分具体是超时、断连、还是干净 EOF
    导致的循环结束——这三种情况的后续处理完全一样。
    """
    while True:
        try:
            chunk = await asyncio.wait_for(content.readany(), timeout=idle_timeout)
        except asyncio.TimeoutError:
            return
        except DISCONNECT_ERRORS:
            return
        if not chunk:
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
    headers = filter_headers(request.headers)
    data = raw_body if raw_body is not None else (request.content if request.can_read_body else None)
    del raw_body  # 用完就扔，见 chat_completions() 里同样处理的注释
    try:
        downstream = await session.request(request.method, target, headers=headers, data=data,
                                          allow_redirects=False)
    except (ClientError, asyncio.TimeoutError) as e:
        return web.Response(status=502, text=f"downstream error: {e}\n")
    del data
    return await _stream_downstream_verbatim(request, downstream)


async def relay(downstream_content, out: web.StreamResponse, state: StreamState,
                 splitter: LineSplitter) -> bool:
    """原始腿：先无限等第一个 chunk——这一层不对"迟迟没有第一个 chunk"这件事负责，等多久、
    要不要重试是上层的事；这期间发生的连接异常也不吞，直接往上抛，让这次请求按普通失败处理，
    不进入续写逻辑（客户端还没看到任何内容，没什么好"救"的）。收到第一个 chunk 之后才开始
    用 idle timeout 盯"卡住"这件事，字节原样转发给客户端（不改写任何内容），同一份字节喂
    SSE 行解析更新 state。返回 True 表示这条腿结束时已经转发过至少一个 chunk、但还没见过
    finish_reason——不管是超时、断连、还是干净 EOF，都算"没说完"，调用方据此决定要不要续写；
    第一个 chunk 就干净 EOF（downstream 一个字节都没吐）时返回 False，不算需要续写。
    """
    chunk = await downstream_content.readany()
    if not chunk:
        return False
    await out.write(chunk)
    for line in splitter.feed(chunk):
        feed_line(line, state)

    async for chunk in timed_reads(downstream_content, config.STALL_IDLE_TIMEOUT_SECONDS):
        await out.write(chunk)
        for line in splitter.feed(chunk):
            feed_line(line, state)
    return state.finish_reason is None


async def relay_leg2(downstream_content, out: web.StreamResponse, splitter: LineSplitter, recovered) -> None:
    """续写腿：逐行转发（不是整块字节透传），因为带 usage 的那条 data 行要原地改写。同样先
    无限等第一个 chunk 再开始用 idle timeout；单次续写策略下，这条腿不管怎么失败（第一个
    chunk 就断、还是后面卡住/断线），都不再发第三条腿，直接安静结束，让调用方把流正常收尾。
    """
    try:
        chunk = await downstream_content.readany()
    except DISCONNECT_ERRORS:
        return
    if not chunk:
        return
    for line in splitter.feed(chunk):
        await out.write(rewrite_usage_line(line, recovered) + b"\n")

    async for chunk in timed_reads(downstream_content, config.STALL_IDLE_TIMEOUT_SECONDS):
        for line in splitter.feed(chunk):
            await out.write(rewrite_usage_line(line, recovered) + b"\n")


async def attempt_continuation(session: ClientSession, target: str, headers: dict, original_payload: dict,
                                state: StreamState, out: web.StreamResponse) -> None:
    recovered = None
    try:
        recovered = await measure_recovered_tokens(session, config.TOKENIZE_URL, original_payload,
                                                     state.reasoning, state.content)
    except Exception:
        log.exception("tokenize measurement failed (known Kimi-K3 /v1/tokenize issue?) — "
                       "continuation proceeds without max_tokens deduction / usage correction: %s", target)

    # 客户端原始请求用的是 max_tokens 还是 max_completion_tokens（新旧两个字段，语义等价），
    # 续写请求就沿用同一个字段名去扣减，不额外发明一个默认预算：客户端两个都没传，意思就是
    # "不限制"，续写腿也不该无中生有地加一个上限。正常客户端只会传其中一个；万一两个都传了
    # （不常见），两个都按扣减后的值改，避免下游到底读哪个字段产生歧义。
    budget_fields = [f for f in ("max_tokens", "max_completion_tokens") if original_payload.get(f) is not None]

    prefix = build_prefix(state.reasoning, state.content)
    continuation_payload = dict(original_payload)
    continuation_payload["messages"] = list(original_payload.get("messages", [])) + [
        {"role": "assistant", "content": prefix}
    ]

    if budget_fields:
        original_budget = original_payload[budget_fields[0]]
        if recovered is not None:
            remaining_budget = original_budget - recovered.total_recovered
            if remaining_budget <= 0:
                log.info("max_tokens budget exhausted before continuation, not retrying: %s", target)
                return
        else:
            # 降级：拿不到已恢复 token 数就不扣减，接受"最多翻倍"的预算风险（只续一次，风险有界）。
            remaining_budget = original_budget
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
        log.exception("continuation leg failed to even start: %s", target)
        return

    if downstream.status != 200:
        log.warning("continuation leg got HTTP %s, giving up: %s", downstream.status, target)
        downstream.release()
        return

    splitter = LineSplitter()
    try:
        await relay_leg2(downstream.content, out, splitter, recovered)
    finally:
        downstream.release()


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
    session: ClientSession = request.app["session"]
    target = config.DOWNSTREAM_URL + request.path_qs
    headers = filter_headers(request.headers)

    try:
        downstream = await session.post(target, headers=headers, data=raw_body, allow_redirects=False)
    except (ClientError, asyncio.TimeoutError) as e:
        return web.Response(status=502, text=f"downstream error: {e}\n")
    # 原始字节只在上面这次 POST 里用一次——续写腿是从解析好的 payload dict 重建的，不需要
    # raw_body。并发多、body 又可能带 base64 图片这种大 payload 时，不早点扔掉这份引用的话，
    # 它会跟着这个协程一直活到整条流结束（可能是几十秒后），白占内存。
    del raw_body

    if downstream.status != 200:
        return await _stream_downstream_verbatim(request, downstream)

    resp_headers = filter_headers(downstream.headers)
    resp_headers.pop("Content-Length", None)
    out = web.StreamResponse(status=downstream.status, headers=resp_headers)
    await out.prepare(request)

    state = StreamState()
    splitter = LineSplitter()
    try:
        needs_retry = await relay(downstream.content, out, state, splitter)
    except (*DISCONNECT_ERRORS, asyncio.CancelledError):
        # 这里同时兜两类情况，都不属于续写范畴、都不吞异常：(1) 下游连接出问题——第一个
        # chunk 都没等到、或者读到一半断了（relay() 内部读下游失败，DISCONNECT_ERRORS）；
        # (2) 客户端（上游）断连——relay() 里往 out 写的时候失败也是同一批 DISCONNECT_ERRORS
        # 类型，不特意区分；如果客户端断连发生在我们正无限等第一个 chunk、还没开始往 out 写
        # 任何东西的时候，靠的是 asyncio.CancelledError——aiohttp 自己的 handler 会在探测到
        # 客户端连接断开时主动 cancel 当前请求的 task，不管这个 task 当时卡在等下游读还是
        # 别的什么地方，不会因为我们这层暂时没有主动的读写操作而漏检。两种情况处理方式一样：
        # 让这次请求按普通失败处理（跟没有这层网关时的行为一致），是否重试交给上层。
        # downstream.close() 确保这个已经坏掉的连接不会被当成好的放回连接池；下面的 finally
        # 还会再调一次 release()，在已经 close() 过的连接上是安全的空操作。
        downstream.close()
        raise
    finally:
        downstream.release()

    if needs_retry:
        if state.tool_calls_seen:
            # v1 范围排除：已经出现过 tool_call chunk，不在网关能安全处理的范围内，不救。
            log.warning("stall after tool_call chunk (out of v1 scope, not retrying): %s", target)
        elif state.reasoning or state.content:
            try:
                await attempt_continuation(session, target, headers, payload, state, out)
            except Exception:
                # 这是网关层，这里再崩会把已经转发给客户端的部分也带没了、甚至可能影响到
                # 同一个 worker 上别的请求；续写本身是"锦上添花"，救不回来不该拖累已经稳妥
                # 转发出去的正文。这里兜的是"意料之外"的错误——tokenize 失败已经在
                # attempt_continuation 内部单独兜过、会降级但不放弃续写，这层是防那些没有
                # 专门处理过的异常（比如构造续写 payload 时的意外类型错误、客户端在续写腿
                # 写入过程中断开连接）。不用担心吞掉 asyncio.CancelledError——Python 3.8+
                # 它是 BaseException 的子类，不会被 `except Exception` 捕获，正常取消/关闭
                # 流程不受影响。
                log.exception("continuation attempt crashed unexpectedly, ending stream as-is: %s",
                               target)
        # else: 收到过 chunk 但没有任何可用的 reasoning/content（比如只有一个空白的角色
        # 声明 chunk），没有实际内容可救，不属于续写范畴，什么都不做，流按原样收尾。

    try:
        await out.write_eof()
    except Exception:
        # 到这一步该做的都做完了，客户端这时候如果已经断开，收尾失败不算真正的错误，
        # 记一下就行，不用再往上抛。
        log.exception("failed to cleanly close the client stream (client likely gone): %s", target)
    return out


async def catch_all(request: web.Request) -> web.StreamResponse:
    return await passthrough(request)


async def on_startup(app: web.Application):
    app["session"] = ClientSession(
        auto_decompress=False,
        timeout=ClientTimeout(total=None, connect=config.CONNECT_TIMEOUT_SECONDS),
    )


async def on_cleanup(app: web.Application):
    await app["session"].close()


def main():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_route("*", "/{tail:.*}", catch_all)
    log.info("listening on 0.0.0.0:%d, forwarding to %s (continuation models: %s)",
              config.PORT, config.DOWNSTREAM_URL, config.CONTINUATION_MODELS or "<none configured>")
    web.run_app(app, host="0.0.0.0", port=config.PORT, print=None)


if __name__ == "__main__":
    main()
