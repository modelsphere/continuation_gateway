"""增量 SSE 行解析——网关边转发原始字节边理解语义，用来判断要不要触发续写。"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamState:
    reasoning_parts: list = field(default_factory=list)
    content_parts: list = field(default_factory=list)
    tool_calls_seen: bool = False
    finish_reason: Optional[str] = None

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)

    @property
    def content(self) -> str:
        return "".join(self.content_parts)


def feed_line(line: bytes, state: StreamState) -> None:
    if not line.startswith(b"data:"):
        return
    data = line[len(b"data:"):].strip()
    if data in (b"[DONE]", b""):
        return
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return
    for ch in obj.get("choices", []):
        delta = ch.get("delta", {})
        if delta.get("reasoning_content"):
            state.reasoning_parts.append(delta["reasoning_content"])
        if delta.get("content"):
            state.content_parts.append(delta["content"])
        if delta.get("tool_calls"):
            state.tool_calls_seen = True
        if ch.get("finish_reason"):
            state.finish_reason = ch["finish_reason"]


class LineSplitter:
    """把任意切法的字节 chunk 重新拼回按 \\n 分割的完整行，跨 chunk 边界缓存半行。"""

    def __init__(self):
        self._buf = b""

    def feed(self, chunk: bytes) -> list:
        self._buf += chunk
        *lines, self._buf = self._buf.split(b"\n")
        return lines
