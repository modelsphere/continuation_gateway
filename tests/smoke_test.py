"""本地冒烟测试：不连真实集群，起一个假 downstream（模拟 SGLang）+ 真的 continuation_gateway
服务，用 aiohttp client 打真实 HTTP 请求，验证网关机制本身对不对：卡住/断连能不能触发续写、
tool_call 出现后是不是老实不救、第一个 chunk 都没等到时是不是彻底不介入（不重试、异常直接
往上传）、usage 改写对不对、按 model 分派的前缀重建（XTML vs 通用 <think> 标签）对不对。
不验证"续写内容语义连不连贯"——那部分要在真实集群上验证，这里只测网关自己写的转发/编排代码。

usage/max_tokens 扣减用到的 token 数不是靠 /v1/tokenize 现测的（这条集成已经被去掉了），是
用 usage.py 的 estimate_tokens() 按字符数估算的，所以下面涉及 token 数的断言直接调用同一个
函数计算期望值，不手工硬编码数字——手工数字容易在 CJK_CHARS_PER_TOKEN/OTHER_CHARS_PER_TOKEN
调整或估算公式变化时悄悄跟代码脱节而不报错。

用法：python tests/smoke_test.py（从 continuation_gateway 仓库根目录下运行，或任意目录下
直接跑这个文件都行，下面会自动把仓库根目录加进 sys.path）。
"""

import asyncio
import json
import os
import sys
import time

os.environ["DOWNSTREAM_URL"] = "http://127.0.0.1:18081"
os.environ["CONTINUATION_MODELS"] = "test-model,kimi-k3"
os.environ["STALL_IDLE_TIMEOUT_SECONDS"] = "0.6"
os.environ["CONNECT_TIMEOUT_SECONDS"] = "5"
os.environ["MAX_CONTINUATION_BODY_MB"] = "1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from aiohttp import ClientPayloadError, ClientSession, ClientTimeout, ServerDisconnectedError, web  # noqa: E402

import continuation_gateway.server as gw_server  # noqa: E402
from continuation_gateway import config as gw_config  # noqa: E402
from continuation_gateway.usage import estimate_tokens  # noqa: E402

DOWNSTREAM_PORT = 18081
GATEWAY_PORT = 18082

call_counts = {}  # scenario -> call count，用来分辨"这是第一条腿还是续写腿"
# upstream_disconnect_before_first_chunk 专用：网关这边的 task 被正确 cancel 会连带断掉
# 它到这个假 downstream 的连接，这里的 asyncio.sleep 会被提前打断而设这个 event，而不是
# 乖乖睡完——用它证明"客户端断连"这个信号真的传导到了网关正在等 leg1 第一个 chunk 的这个
# task，而不只是"客户端自己放弃了"（那个不用靠这个 event 也能验证到）。
downstream_cancel_event = asyncio.Event()


def sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def est(text: str) -> int:
    """跟网关自己算 budget/usage 用的是同一个估算函数，断言直接算期望值，不手工硬编码。"""
    return estimate_tokens(text, gw_config.CJK_CHARS_PER_TOKEN, gw_config.OTHER_CHARS_PER_TOKEN)


async def downstream_chat(request: web.Request) -> web.StreamResponse:
    scenario = request.headers.get("X-Test-Scenario", "default")
    body = await request.json()
    call_counts[scenario] = call_counts.get(scenario, 0) + 1
    call_n = call_counts[scenario]

    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    if scenario == "fault_injection_stall":
        # 只需要 leg1 卡住触发续写尝试，永远不会真的发出 leg2（因为会在 attempt_continuation
        # 内部被注入的故障打断），所以不需要 call_n==2 分支。
        await resp.write(sse({"choices": [{"delta": {"content": "partial before injected fault."}}]}))
        await asyncio.sleep(5)

    elif scenario == "content_done_stall":
        if call_n == 1:
            # leg1 的 completion id——续写发生后客户端应该从头到尾只看到这一个 id，见下面
            # call_n==2 分支里故意分配的不同 id，以及 call_gateway() 里对 ids 的断言。
            await resp.write(sse({"id": "chatcmpl-leg1",
                                   "choices": [{"delta": {"reasoning_content": "let me think. "}}]}))
            await resp.write(sse({"id": "chatcmpl-leg1", "choices": [{"delta": {"content": "The answer is par"}}]}))
            await asyncio.sleep(5)  # 永远等不到下一个 chunk，触发 idle timeout
        else:
            assert body.get("continue_final_message") is True
            # model="test-model" 走的是通用 <think>...</think> builder（非 kimi-k3 的默认
            # 分支，见 reconstruct.py _BUILDERS），XTML 分派单独在 kimi_k3_xtml_stall 场景测。
            assert "</think>The answer is par" in body["messages"][-1]["content"]
            expected_budget = 1000 - (est("let me think. ") + est("The answer is par"))
            assert body.get("max_tokens") == expected_budget, \
                f"max_tokens={body.get('max_tokens')}, expected={expected_budget}"
            # 故意跟 leg1 的 id 不一样——真实下游对续写请求（网关自己发起的新 HTTP 请求）会
            # 分配一个全新的 completion id，网关必须把它改写回 leg1 的原始 id 再转发给客户端。
            leg2_id = "chatcmpl-leg2-should-be-rewritten"
            await resp.write(sse({"id": leg2_id, "choices": [{"delta": {"content": "tial recovery."}}]}))
            await resp.write(sse({"id": leg2_id, "choices": [{"delta": {}, "finish_reason": "stop"}]}))
            await resp.write(sse({"id": leg2_id, "choices": [], "usage": {
                "prompt_tokens": 999, "completion_tokens": 3, "total_tokens": 1002, "reasoning_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 5000},
            }}))
            await resp.write(b"data: [DONE]\n\n")

    elif scenario == "thinking_partial_disconnect":
        if call_n == 1:
            await resp.write(sse({"choices": [{"delta": {"reasoning_content": "step one. "}}]}))
            await resp.write(sse({"choices": [{"delta": {"reasoning_content": "step two"}}]}))
            await resp.write_eof()
            resp.force_close()
            raise ConnectionResetError("simulated crash mid-stream")
        else:
            assert body.get("continue_final_message") is True
            # model="test-model" 走通用 <think> builder，content 为空所以不闭合标签
            # （见 reconstruct.py _build_prefix_think_tag 的说明）。
            assert body["messages"][-1]["content"] == "<think>step one. step two"
            assert body.get("chat_template_kwargs") is None  # thinking 不该被关
            await resp.write(sse({"choices": [{"delta": {"reasoning_content": " step three."}}]}))
            await resp.write(sse({"choices": [{"delta": {"content": "done."}}]}))
            await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
            await resp.write(sse({"choices": [], "usage": {
                "prompt_tokens": 500, "completion_tokens": 5, "total_tokens": 505, "reasoning_tokens": 3,
            }}))
            await resp.write(b"data: [DONE]\n\n")

    elif scenario == "kimi_k3_xtml_stall":
        # 专门验证 reconstruct.py 的按 model 分派：model="kimi-k3" 应该走 XTML builder
        # （<|open|>think<|sep|>...<|close|>think<|sep|><|open|>response<|sep|>...），
        # 跟 test-model 走的通用 <think> 格式不是同一套。
        if call_n == 1:
            await resp.write(sse({"choices": [{"delta": {"reasoning_content": "k3 thinking. "}}]}))
            await resp.write(sse({"choices": [{"delta": {"content": "k3 answer"}}]}))
            await asyncio.sleep(5)
        else:
            assert body.get("continue_final_message") is True
            assert body["messages"][-1]["content"] == (
                "<|open|>think<|sep|>k3 thinking. <|close|>think<|sep|><|open|>response<|sep|>k3 answer"
            ), body["messages"][-1]["content"]
            await resp.write(sse({"choices": [{"delta": {"content": " continued."}}]}))
            await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
            await resp.write(sse({"choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "reasoning_tokens": 0,
            }}))
            await resp.write(b"data: [DONE]\n\n")

    elif scenario == "tool_call_seen_no_rescue":
        await resp.write(sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "x:0", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ]}}]}))
        await asyncio.sleep(5)  # 卡住，但 v1 范围排除 tool_call，网关不该发第二条腿
        assert call_n == 1, "不该有第二次调用"

    elif scenario == "no_chunk_clean_eof":
        # 一个字节都没吐、直接干净 EOF——不属于续写范畴，网关不该重试，只应该有这一次调用。
        assert call_n == 1, "不该有第二次调用"

    elif scenario == "no_chunk_disconnect":
        # 一个字节都没吐就直接断连——同样不属于续写范畴，异常应该原样往上传，不能被吞掉
        # 变成一个"正常但空"的 200 响应。
        assert call_n == 1, "不该有第二次调用"
        raise ConnectionResetError("simulated crash before any chunk")

    elif scenario == "upstream_disconnect_before_first_chunk":
        # 这个假 downstream 已经连上、响应头也发了（resp.prepare() 在函数最上面已经调过），
        # 但故意长时间不写任何 body 字节——模拟"leg1 已经连上但迟迟没有第一个 chunk"，网关
        # 这时候正卡在 relay() 里 `await downstream_content.readany()`。真正要验证的不是
        # 下游这边的行为，是网关那边：客户端提前断连之后，网关是不是把这个还在等第一个
        # chunk 的 task 尽快 cancel 掉了（handler_cancellation=True 保护的正是这个阶段，
        # 默认关闭时网关会傻等到自己的超时才反应）。用 CancelledError
        # 而不是等 sleep 自然结束来判断"网关是不是真的提前断了"——网关那边的 task 被正确
        # cancel，会连带断掉它这一端到这个假 downstream 的连接，这里的 sleep 会被打断。
        try:
            await asyncio.sleep(8)
        except asyncio.CancelledError:
            downstream_cancel_event.set()
            raise
        return web.Response(status=200, text="should never get here")

    elif scenario == "passthrough_check":
        # 纯透传场景：不管调几次都立刻正常返回，不卡、不断连，用来验证非 continuation
        # model 走的是纯转发，网关完全没有介入（没有 idle timeout 包装、没有 SSE 解析）。
        await resp.write(sse({"choices": [{"delta": {"content": "untouched passthrough."}}]}))
        await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        await resp.write(b"data: [DONE]\n\n")

    elif scenario == "oversized_body_passthrough":
        # body 超过 MAX_CONTINUATION_BODY_MB：即使 model/stream 都满足续写条件，也应该
        # 被当成普通请求纯转发——立刻正常返回，不卡、不断连，用来确认没有被 guarded() 接管。
        await resp.write(sse({"choices": [{"delta": {"content": "oversized, untouched."}}]}))
        await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        await resp.write(b"data: [DONE]\n\n")

    elif scenario == "response_format_passthrough":
        # response_format 是 json_object/json_schema：即使 model/stream/body 都满足续写
        # 条件，也应该被当成普通请求纯转发——立刻正常返回，不卡、不断连，用来确认没有被
        # guarded() 接管。
        await resp.write(sse({"choices": [{"delta": {"content": '{"answer": "untouched"}'}}]}))
        await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        await resp.write(b"data: [DONE]\n\n")

    elif scenario == "multimodal_passthrough":
        # messages 里带了 image_url 这种非文本 content-part，故意卡住比 idle timeout 更久
        # （0.6s）——如果这条请求被 guarded() 接管了，会触发续写、发出第二次下游调用；
        # 断言 call_n == 1 就是在证明它压根没被接管，是纯转发（下游卡多久网关都不管），不是
        # "碰巧这次没卡住所以看不出区别"那种弱验证。estimate_tokens() 对非文本内容没有对应
        # 的字符数可数，续写发生的话 corrected prompt_tokens 会严重偏小，属于账目算不准，
        # 不是"代价不划算"，所以直接排除，不进 guarded()。
        await resp.write(sse({"choices": [{"delta": {"content": "described the image partial"}}]}))
        await asyncio.sleep(5)
        assert call_n == 1, "不该有第二次调用（多模态不进续写）"

    elif scenario == "no_budget_field":
        if call_n == 1:
            await resp.write(sse({"choices": [{"delta": {"content": "no limit set"}}]}))
            await asyncio.sleep(5)
        else:
            assert "max_tokens" not in body, f"不该凭空加 max_tokens: {body}"
            assert "max_completion_tokens" not in body, f"不该凭空加 max_completion_tokens: {body}"
            await resp.write(sse({"choices": [{"delta": {"content": " continued."}}]}))
            await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
            await resp.write(sse({"choices": [], "usage": {
                "prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52, "reasoning_tokens": 0,
            }}))
            await resp.write(b"data: [DONE]\n\n")

    elif scenario == "tool_call_seen_stall_then_resumes":
        # 验证"tool_call 已出现后 idle timeout 不再结束这条腿"：卡住的时间比 idle timeout
        # (0.6s) 长，但下游其实没死，之后还会正常吐完。如果网关在 tool_calls_seen 之后仍然
        # 让 idle timeout 结束循环，这条流会在 finish_reason 到达之前就被切断，下面的断言会
        # 直接失败；只有网关老实等下去才能收到完整的 finish_reason。
        await resp.write(sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "x:0", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ]}}]}))
        await asyncio.sleep(0.9)
        await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
        await resp.write(b"data: [DONE]\n\n")
        assert call_n == 1, "不该有第二次调用（tool_call 出现后不救）"

    elif scenario == "no_content_stall_then_resumes":
        # 验证"还没攒到任何可恢复内容时 idle timeout 也不该结束这条腿"：只吐了一个空白的
        # 角色声明 chunk（reasoning/content 都是空的），卡住的时间比 idle timeout 长，之后
        # 下游继续正常吐出实质内容。如果网关在"目前没有可恢复内容"时仍然让 idle timeout
        # 结束循环（旧行为），这条流会在任何实质内容到达之前就被切断收尾。
        await resp.write(sse({"choices": [{"delta": {"role": "assistant"}}]}))
        await asyncio.sleep(0.9)
        await resp.write(sse({"choices": [{"delta": {"content": "actually here"}}]}))
        await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        await resp.write(b"data: [DONE]\n\n")
        assert call_n == 1, "不该有第二次调用（不是续写场景，只是老实等）"

    elif scenario == "leg2_stall_then_resumes":
        # 验证 leg2 同样的规则：单次续写没有第三条腿可退，所以 leg2 里的 idle timeout 也
        # 不该结束这条腿——卡住的时间比 idle timeout 长，之后 leg2 还会正常吐完。如果网关
        # 在 leg2 卡住时就放弃收尾（旧行为），最终 content/finish_reason 收不全。
        if call_n == 1:
            await resp.write(sse({"choices": [{"delta": {"content": "leg1 partial"}}]}))
            await asyncio.sleep(5)
        else:
            assert body.get("continue_final_message") is True
            await resp.write(sse({"choices": [{"delta": {"content": " leg2 first half"}}]}))
            await asyncio.sleep(0.9)
            await resp.write(sse({"choices": [{"delta": {"content": " leg2 second half"}}]}))
            await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
            await resp.write(sse({"choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "reasoning_tokens": 0,
            }}))
            await resp.write(b"data: [DONE]\n\n")

    elif scenario == "max_completion_tokens_field":
        if call_n == 1:
            await resp.write(sse({"choices": [{"delta": {"content": "partial with mct"}}]}))
            await asyncio.sleep(5)
        else:
            assert "max_tokens" not in body, f"不该动客户端没传的 max_tokens: {body}"
            expected_budget = 1000 - est("partial with mct")
            assert body.get("max_completion_tokens") == expected_budget, \
                f"max_completion_tokens={body.get('max_completion_tokens')}, expected={expected_budget}"
            await resp.write(sse({"choices": [{"delta": {"content": " continued."}}]}))
            await resp.write(sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
            await resp.write(sse({"choices": [], "usage": {
                "prompt_tokens": 60, "completion_tokens": 2, "total_tokens": 62, "reasoning_tokens": 0,
            }}))
            await resp.write(b"data: [DONE]\n\n")

    await resp.write_eof()
    return resp


async def run_downstream():
    app = web.Application(client_max_size=gw_config.MAX_REQUEST_BODY_BYTES)
    app.router.add_post("/v1/chat/completions", downstream_chat)
    # handler_cancellation=True：跟 run_gateway() 一样的原因（见那边注释）。这个假 downstream
    # 自己不需要靠这个感知谁断了它，但 upstream_disconnect_before_first_chunk 那个场景要靠
    # "网关断开自己到这个假 downstream 的连接后，这里的 sleep 被 CancelledError 打断"来
    # 证明网关那边真的提前 cancel 了——如果这里不开，就算网关那边修好了，这个假 downstream
    # 的 handler 也感知不到连接已经没了，会乖乖睡完，测试会看起来像没修好。
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", DOWNSTREAM_PORT)
    await site.start()
    return runner


async def run_gateway():
    app = web.Application(client_max_size=gw_config.MAX_REQUEST_BODY_BYTES)
    app.on_startup.append(gw_server.on_startup)
    app.on_cleanup.append(gw_server.on_cleanup)
    app.router.add_post("/v1/chat/completions", gw_server.chat_completions)
    app.router.add_get("/health", gw_server.health)
    app.router.add_route("*", "/{tail:.*}", gw_server.catch_all)
    # handler_cancellation=True：镜像 server.py:main() 里 web.run_app() 的同一个参数（见
    # 那边注释）——不带这个参数会用 aiohttp 的默认值 False，这个测试服务器的行为就跟生产
    # 环境不一样了。这里手搭的 web.AppRunner 和 aiohttp.test_utils（它默认就是
    # handler_cancellation=True）走的是两条默认值不同的路径，用后者测出来的"客户端断连
    # 不管卡在哪都能被感知到"这类结论不能代表用前者搭建的生产服务的真实行为。
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", GATEWAY_PORT)
    await site.start()
    return runner


async def call_gateway(scenario: str, model: str = "test-model", extra: dict = None, drop: list = None):
    payload = {"model": model, "stream": True, "max_tokens": 1000, "temperature": 0.7,
               "messages": [{"role": "user", "content": "hi"}]}
    for key in drop or []:
        payload.pop(key, None)
    if extra:
        payload.update(extra)
    reasoning, content, finish_reason, usage, ids = [], [], None, None, []
    aborted = False
    async with ClientSession(timeout=ClientTimeout(total=15)) as client:
        async with client.post(f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
                                json=payload, headers={"X-Test-Scenario": scenario}) as resp:
            status = resp.status
            try:
                async for raw_line in resp.content:
                    line = raw_line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if obj.get("id"):
                        ids.append(obj["id"])
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for ch in obj.get("choices", []):
                        d = ch.get("delta", {})
                        if d.get("reasoning_content"):
                            reasoning.append(d["reasoning_content"])
                        if d.get("content"):
                            content.append(d["content"])
                        if ch.get("finish_reason"):
                            finish_reason = ch["finish_reason"]
            except (ClientPayloadError, ServerDisconnectedError, ConnectionResetError):
                # 网关侧异常应该原样往上传，客户端这里应该真的看到连接被异常中断——
                # 这正是 no_chunk_disconnect 场景要验证的行为，不是测试的 bug。
                aborted = True
    return {"status": status, "reasoning": "".join(reasoning), "content": "".join(content),
            "finish_reason": finish_reason, "usage": usage, "aborted": aborted, "ids": ids}


async def main():
    downstream_runner = await run_downstream()
    gateway_runner = await run_gateway()
    failures = []

    def check(name, cond, detail=""):
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {name} {detail}")
        if not cond:
            failures.append(name)

    try:
        print("== content_done_stall (idle timeout, usage 改写) ==")
        r = await call_gateway("content_done_stall")
        print("  ", r)
        check("content merged correctly", r["content"] == "The answer is partial recovery.", r["content"])
        check("finish_reason present", r["finish_reason"] == "stop")
        expected_prompt_tokens = est("hi")  # 原始请求 messages 里唯一的文本
        check("usage prompt_tokens corrected (char-estimated)",
              r["usage"]["prompt_tokens"] == expected_prompt_tokens, r["usage"])
        # completion_tokens 是"全部生成 token 数"（含 reasoning，SGLang 源码确认），所以要把
        # 被救回的 reasoning 和 content 都加上，不能只加 content。
        expected_completion = 3 + est("let me think. ") + est("The answer is par")
        check("usage completion_tokens = leg2(3) + recovered reasoning + recovered content",
              r["usage"]["completion_tokens"] == expected_completion, r["usage"])
        check("cached_tokens clamped to <= corrected prompt_tokens",
              r["usage"]["prompt_tokens_details"]["cached_tokens"] <= r["usage"]["prompt_tokens"], r["usage"])
        # leg2 的 mock 下游把 id 换成了 "chatcmpl-leg2-should-be-rewritten"（模拟真实下游给
        # 续写请求分配的新 completion id），网关必须把它改写回 leg1 的原始 id，客户端才不会
        # 在同一个响应里看到 id 中途变化。
        check("id stays the same across leg1/leg2 (rewritten back to leg1's original id)",
              r["ids"] and all(i == "chatcmpl-leg1" for i in r["ids"]), r["ids"])

        print("\n== thinking_partial_disconnect (断连触发 + thinking 不能被关) ==")
        r = await call_gateway("thinking_partial_disconnect")
        print("  ", r)
        check("reasoning merged correctly", r["reasoning"] == "step one. step two step three.", r["reasoning"])
        check("content from leg2 present", r["content"] == "done.", r["content"])
        check("finish_reason present", r["finish_reason"] == "stop")
        # 崩溃时 content 还没开始，recovered.content_tokens=0；recovered reasoning 仍然要计入
        # completion_tokens，不能因为 content 是空的就漏算。
        recovered_reasoning = est("step one. step two")
        check("usage completion_tokens = leg2(5) + recovered reasoning + recovered content(0)",
              r["usage"]["completion_tokens"] == 5 + recovered_reasoning + 0, r["usage"])
        check("usage reasoning_tokens = leg2(3) + recovered reasoning",
              r["usage"]["reasoning_tokens"] == 3 + recovered_reasoning, r["usage"])

        print("\n== kimi_k3_xtml_stall (model=kimi-k3 走 XTML builder, 跟通用 <think> 格式不同) ==")
        r = await call_gateway("kimi_k3_xtml_stall", model="kimi-k3")
        print("  ", r)
        check("content merged correctly", r["content"] == "k3 answer continued.", r["content"])
        check("finish_reason present", r["finish_reason"] == "stop")

        print("\n== tool_call_seen_no_rescue (v1 范围排除，不救) ==")
        r = await call_gateway("tool_call_seen_no_rescue")
        print("  ", r)
        check("no finish_reason (流被卡住后老实结束，不续写)", r["finish_reason"] is None)
        check("only 1 downstream call made", call_counts.get("tool_call_seen_no_rescue") == 1)

        print("\n== multimodal_passthrough (带图片的请求不该进续写，账不出对) ==")
        r = await call_gateway("multimodal_passthrough", extra={"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,fakeimagedata"}},
            ]},
        ]})
        print("  ", r)
        check("no finish_reason (流被卡住后老实结束，不续写)", r["finish_reason"] is None)
        check("only 1 downstream call made (没有触发续写编排)",
              call_counts.get("multimodal_passthrough") == 1)

        print("\n== tool_call_seen_stall_then_resumes (tool_call 出现后 idle timeout 不该掐断连接) ==")
        r = await call_gateway("tool_call_seen_stall_then_resumes")
        print("  ", r)
        check("finish_reason eventually arrives (没有被 idle timeout 提前掐断)",
              r["finish_reason"] == "tool_calls", r["finish_reason"])
        check("only 1 downstream call made (仍然没有触发续写)",
              call_counts.get("tool_call_seen_stall_then_resumes") == 1)

        print("\n== no_content_stall_then_resumes (还没有可恢复内容时 idle timeout 不该掐断连接) ==")
        r = await call_gateway("no_content_stall_then_resumes")
        print("  ", r)
        check("content eventually arrives (没有被 idle timeout 提前掐断)",
              r["content"] == "actually here", r["content"])
        check("finish_reason present", r["finish_reason"] == "stop")
        check("only 1 downstream call made (老实等到底，不是靠续写救回来的)",
              call_counts.get("no_content_stall_then_resumes") == 1)

        print("\n== leg2_stall_then_resumes (leg2 里 idle timeout 也不该掐断连接) ==")
        r = await call_gateway("leg2_stall_then_resumes")
        print("  ", r)
        check("leg2 content merged across the stall",
              r["content"] == "leg1 partial leg2 first half leg2 second half", r["content"])
        check("finish_reason present (leg2 卡住后没有被提前收尾)", r["finish_reason"] == "stop")
        check("only 2 downstream calls (leg1 + leg2, 没有第三条腿)",
              call_counts.get("leg2_stall_then_resumes") == 2)

        print("\n== no_chunk_clean_eof (一个字节都没吐、干净 EOF，不属于续写范畴) ==")
        r = await call_gateway("no_chunk_clean_eof")
        print("  ", r)
        check("no content", r["content"] == "", r["content"])
        check("no finish_reason", r["finish_reason"] is None)
        check("not aborted (这是干净的空响应，不是异常)", not r["aborted"])
        check("only 1 downstream call (没有重试)", call_counts.get("no_chunk_clean_eof") == 1)

        print("\n== no_chunk_disconnect (一个字节都没吐就断连，异常应该原样往上传) ==")
        r = await call_gateway("no_chunk_disconnect")
        print("  ", r)
        check("client sees the connection abort (not silently swallowed into empty 200)",
              r["aborted"])
        check("only 1 downstream call (没有重试)", call_counts.get("no_chunk_disconnect") == 1)

        print("\n== upstream_disconnect_before_first_chunk (客户端在等第一个 chunk 阶段先"
              "断连，网关不该傻等) ==")
        # 客户端自己配一个很短的 total 超时（1s），远小于下游这次故意挂着不回的时长（8s），
        # 模拟"客户端超时早于下游能给出任何响应"这种结构——客户端自己配的超时明显短于网关
        # 等下游的耗时上限时，不该是网关先撑到自己的超时才反应，应该是客户端一放弃网关就
        # 跟着反应过来。
        downstream_cancel_event.clear()
        t0 = time.monotonic()
        client_gave_up = False
        try:
            async with ClientSession(timeout=ClientTimeout(total=1)) as client:
                async with client.post(
                        f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
                        json={"model": "test-model", "stream": True, "max_tokens": 100,
                              "messages": [{"role": "user", "content": "hi"}]},
                        headers={"X-Test-Scenario": "upstream_disconnect_before_first_chunk"}) as resp:
                    await resp.read()
        except asyncio.TimeoutError:
            client_gave_up = True
        check("client gave up around its own 1s timeout", client_gave_up)
        try:
            # 给网关一点反应时间（远小于下游 8s 的 sleep），但不需要等太久——修复生效的话
            # 应该几乎立刻（毫秒级）就传导到位，3s 已经是很宽松的上限。
            await asyncio.wait_for(downstream_cancel_event.wait(), timeout=3.0)
            gateway_reacted_fast = True
        except asyncio.TimeoutError:
            gateway_reacted_fast = False
        elapsed = time.monotonic() - t0
        check("gateway aborted its own leg1 wait promptly after client gave up "
              "(not waiting for downstream's full 8s hang)", gateway_reacted_fast,
              f"elapsed={elapsed:.2f}s")
        check("only 1 downstream call",
              call_counts.get("upstream_disconnect_before_first_chunk") == 1)

        print("\n== no_budget_field (客户端两个字段都没传，续写请求不该凭空加预算) ==")
        r = await call_gateway("no_budget_field", drop=["max_tokens"])
        print("  ", r)
        check("content merged correctly", r["content"] == "no limit set continued.", r["content"])
        check("finish_reason present", r["finish_reason"] == "stop")

        print("\n== max_completion_tokens_field (客户端传的是新字段名，续写要跟着改这个字段) ==")
        r = await call_gateway("max_completion_tokens_field", drop=["max_tokens"],
                                extra={"max_completion_tokens": 1000})
        print("  ", r)
        check("content merged correctly", r["content"] == "partial with mct continued.", r["content"])
        check("finish_reason present", r["finish_reason"] == "stop")

        print("\n== oversized_body_passthrough (body 超过 MAX_CONTINUATION_BODY_MB=1) ==")
        # aiohttp web.Application 默认 client_max_size 是 2MB，padding 必须超过
        # MAX_CONTINUATION_BODY_MB=1 但留在这个硬上限以内，不然请求会在到达网关业务逻辑之前
        # 就被 aiohttp 自己拒绝（跟这里想测的"续写阈值排除"是两回事）。
        r = await call_gateway("oversized_body_passthrough", extra={"padding": "x" * 1_200_000})
        print("  ", r)
        check("content from downstream verbatim (走的是 passthrough)",
              r["content"] == "oversized, untouched.", r["content"])
        check("finish_reason present", r["finish_reason"] == "stop")
        check("only 1 downstream call (没有触发续写编排)",
              call_counts.get("oversized_body_passthrough") == 1)

        for rf in [{"type": "json_object"}, {"type": "json_schema", "json_schema": {"name": "x"}}]:
            label = rf["type"]
            print(f"\n== response_format={label} (结构化输出不进续写) ==")
            r = await call_gateway("response_format_passthrough", extra={"response_format": rf})
            print("  ", r)
            check(f"[{label}] content from downstream verbatim (走的是 passthrough)",
                  r["content"] == '{"answer": "untouched"}', r["content"])
            check(f"[{label}] finish_reason present", r["finish_reason"] == "stop")
        check("exactly 1 downstream call per response_format case, 2 total (没有触发续写编排)",
              call_counts.get("response_format_passthrough") == 2)

        print("\n== fault_injection_stall (续写阶段内部意外抛异常，网关不能崩) ==")

        def _boom(model, reasoning, content):
            raise RuntimeError("simulated unexpected bug inside the continuation phase")

        original_build_prefix = gw_server.build_prefix
        gw_server.build_prefix = _boom
        try:
            r = await call_gateway("fault_injection_stall")
        finally:
            gw_server.build_prefix = original_build_prefix
        print("  ", r)
        check("client still gets leg1's partial content (not aborted)",
              r["content"] == "partial before injected fault.", r["content"])
        check("no finish_reason (rescue never completed)", r["finish_reason"] is None)
        check("connection not aborted to the client (异常被兜住了，不是原样往上传)",
              not r["aborted"])
        check("only 1 downstream chat call (从没成功发出续写请求)",
              call_counts.get("fault_injection_stall") == 1)

        print("\n== 非 continuation model：纯透传，不触发任何续写逻辑 ==")
        r = await call_gateway("passthrough_check", model="some-other-model")
        print("  ", r)
        check("passthrough got downstream content verbatim", r["content"] == "untouched passthrough.", r["content"])
        check("only 1 downstream call (no retry machinery engaged)",
              call_counts.get("passthrough_check") == 1)

        print("\n== /health (网关自己的存活探测，不该落进 catch_all 透传给下游) ==")
        # downstream 这个假 app 没有注册 /health 路由——如果网关的 /health 不小心走进了
        # catch_all/passthrough，这里会看到下游返回的 404，而不是网关自己的 {"status": "ok"}。
        async with ClientSession(timeout=ClientTimeout(total=5)) as client:
            async with client.get(f"http://127.0.0.1:{GATEWAY_PORT}/health") as resp:
                health_status = resp.status
                health_body = await resp.json()
        print("  ", health_status, health_body)
        check("status 200", health_status == 200, health_status)
        check("body reports gateway's own status, not proxied to downstream",
              health_body == {"status": "ok"}, health_body)

    finally:
        await gateway_runner.cleanup()
        await downstream_runner.cleanup()

    print(f"\n{'='*60}\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
