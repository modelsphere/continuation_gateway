"""增量 SSE 行解析——网关边转发原始字节边理解语义，用来判断要不要触发续写。"""

import json
from dataclasses import dataclass, field
from typing import Optional


# 思考内容的字段名，SGLang 用 reasoning_content，vLLM 用 reasoning，同一个响应里只会出现其中
# 一个。
REASONING_FIELDS = ("reasoning_content", "reasoning")


@dataclass
class StreamState:
    reasoning_parts: list = field(default_factory=list)
    content_parts: list = field(default_factory=list)
    tool_calls_seen: bool = False
    finish_reason: Optional[str] = None
    # leg1 第一条 data 行里的 completion id（如 "chatcmpl-xxx"）。只记第一次见到的值——
    # 续写腿是网关自己发起的第二个下游请求，下游会给它分配一个全新的 id，如果直接透传，
    # 客户端会在同一个响应流里看到 id 中途变了。这个字段记下来是为了在续写腿把 id 改写回
    # 这个原始值，见 usage.py 的 rewrite_leg2_line；同时也是排查问题时网关日志和客户端
    # 抓包能对上的唯一自然 key（客户端本来就能从收到的每个 chunk 里读到这个 id，不需要
    # 网关另外发明一个只有日志里才看得到的标识符）。
    response_id: Optional[str] = None
    # leg1 里思考内容用的字段名：SGLang 是 "reasoning_content"，vLLM 是 "reasoning"，只记第一
    # 次见到非空思考文本时用的那个。续写腿如果被路由到另一种后端，字段名可能跟 leg1 不同，
    # usage.py 的 rewrite_leg2_line 据此把续写腿的思考字段改写成这个名字，客户端看到的流
    # 里字段名前后一致。leg1 没出现过思考内容时保持 None，续写腿的字段名不改写。
    reasoning_field: Optional[str] = None
    # BUFFER_TOOL_CALLS 策略下用到（见 feed_line_holding）：tool_call 一开始就把这一行和之后的
    # 所有行暂存在 held_lines 里，到 finish_reason/[DONE] 才一起放行；此期间 tool_calls_seen
    # 保持 False（客户端还没收到过任何 tool_call），reasoning/content 也不再累积，这样
    # 崩溃时 state 就精确停在"第一个 tool_call chunk 之前"的状态，可以直接按普通的
    # thinking-partial/content-done 续写。
    holding_tool_calls: bool = False
    held_lines: list = field(default_factory=list)

    def discard_held_tool_calls(self) -> tuple:
        """丢弃暂存的 tool_call 行，回到"从没见过 tool_call"的状态，返回 (行数, 字节数)。"""
        lines, size = len(self.held_lines), sum(len(l) + 1 for l in self.held_lines)
        self.held_lines = []
        self.holding_tool_calls = False
        return lines, size

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)

    @property
    def content(self) -> str:
        return "".join(self.content_parts)


def _parse_data_line(line: bytes) -> Optional[dict]:
    if not line.startswith(b"data:"):
        return None
    data = line[len(b"data:"):].strip()
    if data in (b"[DONE]", b""):
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _has_tool_calls(obj: dict) -> bool:
    return any((ch.get("delta") or {}).get("tool_calls") for ch in obj.get("choices", []))


def _apply(obj: dict, state: StreamState, text_and_tool_calls: bool = True) -> None:
    """text_and_tool_calls=False 时只记 id/finish_reason，不累积 reasoning/content、不置
    tool_calls_seen——暂存 tool_call 期间用，见 feed_line_holding()。"""
    if state.response_id is None and obj.get("id"):
        state.response_id = obj["id"]
    for ch in obj.get("choices", []):
        delta = ch.get("delta", {})
        if text_and_tool_calls:
            for name in REASONING_FIELDS:
                if delta.get(name):
                    state.reasoning_parts.append(delta[name])
                    if state.reasoning_field is None:
                        state.reasoning_field = name
                    break
            if delta.get("content"):
                state.content_parts.append(delta["content"])
            if delta.get("tool_calls"):
                state.tool_calls_seen = True
        if ch.get("finish_reason"):
            state.finish_reason = ch["finish_reason"]


def feed_line(line: bytes, state: StreamState) -> None:
    obj = _parse_data_line(line)
    if obj is not None:
        _apply(obj, state)


def feed_line_holding(line: bytes, state: StreamState) -> list:
    """BUFFER_TOOL_CALLS 策略下的逐行处理，返回"现在就该转发给客户端的行"（不含行尾换行）。

    没见到 tool_call 之前跟 feed_line() 一样正常更新 state、原样放行。第一个带 tool_calls
    的行出现后进入暂存：这一行和之后的所有行（含分隔的空行）都进 state.held_lines，不转发，
    也不更新 reasoning/content/tool_calls_seen，只记 finish_reason/id。收到 finish_reason
    （tool_call 已经全部吐完）或 [DONE]（流已经被下游宣告结束，此时不管有没有 finish_reason
    都不该再把暂存内容扣下）时，整批暂存内容连同当前行一起返回，并置 tool_calls_seen——
    客户端这时才真正收到了 tool_call，之后再崩就不属于可续写的范围。如果在这之前流断了，
    暂存内容留在 state.held_lines 里，由调用方决定丢弃（去续写）还是原样放出。
    """
    obj = _parse_data_line(line)
    is_done = line.startswith(b"data:") and line[len(b"data:"):].strip() == b"[DONE]"
    if not state.holding_tool_calls:
        if obj is None or not _has_tool_calls(obj):
            if obj is not None:
                _apply(obj, state)
            return [line]
        state.holding_tool_calls = True
    state.held_lines.append(line)
    if obj is not None:
        _apply(obj, state, text_and_tool_calls=False)
    if state.finish_reason is None and not is_done:
        return []
    released, state.held_lines = state.held_lines, []
    state.holding_tool_calls = False
    state.tool_calls_seen = True
    return released


class LineSplitter:
    """把任意切法的字节 chunk 重新拼回按 \\n 分割的完整行，跨 chunk 边界缓存半行。"""

    def __init__(self):
        self._buf = b""

    def feed(self, chunk: bytes) -> list:
        self._buf += chunk
        *lines, self._buf = self._buf.split(b"\n")
        return lines
