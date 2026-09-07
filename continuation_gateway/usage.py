"""usage/账单修正。

崩溃的那条腿由于连接中断，永远不会上报 usage；续写这条腿自己的 usage 是"这条腿单独生成了
多少"，不包含被崩溃吞掉、又靠续写救回来的那部分内容——如果直接把续写腿的 usage 转发给客户端，
会看到 completion_tokens 异常小、看起来像是内容丢了。这里的公式只做加法，不追求绝对精确：
`corrected_prompt_tokens`/`corrected_completion/reasoning_tokens` 里"被救回的那部分"用字符数
估算出 token 数，加到续写腿真实上报的 usage 上。

token 数不再靠 /v1/tokenize 现测（下游不想为这个改服务，且 messages 模式在 Kimi-K3 部署上
一直有已知问题），改成 estimate_tokens() 按字符数估算：CJK（中/日/韩）字符和其它字符分开算，
用两个经验比例（见 config.py 的 CJK_CHARS_PER_TOKEN / OTHER_CHARS_PER_TOKEN）换算成 token 数。
这两个比例是通用经验值，不是针对 Kimi-K3 分词器实测校准过的，估算注定比真实分词有偏差——
但这层本来就"只做加法、不追求绝对精确"，偏差是可接受的。
"""

import json
import math
import re
from typing import Optional

_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힣豈-﫿]")


class RecoveredTokens:
    __slots__ = ("prompt_tokens", "reasoning_tokens", "content_tokens")

    def __init__(self, prompt_tokens: int, reasoning_tokens: int, content_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.reasoning_tokens = reasoning_tokens
        self.content_tokens = content_tokens

    @property
    def total_recovered(self) -> int:
        return self.reasoning_tokens + self.content_tokens


def estimate_tokens(text: str, cjk_chars_per_token: float, other_chars_per_token: float) -> int:
    """按字符数估算 token 数。CJK 字符在大多数 BPE 分词器里比其它语言编码得密得多，混在一起
    按同一个比例算误差会很大，所以分开数、分开换算再相加。"""
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars
    return math.ceil(cjk_chars / cjk_chars_per_token + other_chars / other_chars_per_token)


def _message_text(message: dict) -> str:
    parts = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        # 多模态 content-parts 格式：只收文本部分，image_url 之类的不参与估算（本来就没有
        # 直接对应的字符数可数，这类大 body 请求也基本会被 MAX_CONTINUATION_BODY_MB 挡在
        # 续写范畴外）。
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        parts.append(function.get("name", ""))
        parts.append(function.get("arguments", ""))
    return "".join(parts)


def _prompt_text(payload: dict) -> str:
    """把原始请求里跟 prompt token 数相关的文本拼成一整段——messages 的正文/tool_calls 参数，
    加上 tools 的 schema（agent 场景里 tools 定义本身往往占不小的 token 量，不能漏）。不追求
    跟 chat_template 渲染出来的文本逐字节一致（role 分隔符之类的控制 token 数量小，这里索性
    忽略，跟"按字符估算"本身的精度取舍一致），只把实际内容的字符量尽量收全。"""
    texts = [_message_text(m) for m in payload.get("messages", [])]
    tools = payload.get("tools")
    if tools:
        texts.append(json.dumps(tools, ensure_ascii=False))
    return "".join(texts)


def estimate_recovered_tokens(original_payload: dict, reasoning_text: str, content_text: str,
                               cjk_chars_per_token: float, other_chars_per_token: float) -> RecoveredTokens:
    prompt_tokens = estimate_tokens(_prompt_text(original_payload), cjk_chars_per_token, other_chars_per_token)
    reasoning_tokens = estimate_tokens(reasoning_text, cjk_chars_per_token, other_chars_per_token)
    content_tokens = estimate_tokens(content_text, cjk_chars_per_token, other_chars_per_token)
    return RecoveredTokens(prompt_tokens, reasoning_tokens, content_tokens)


def correct_usage(recovered: RecoveredTokens, final_leg_usage: dict) -> dict:
    """只做加法，不从 prompt_tokens 里减——从 prompt_tokens 里减需要知道"崩溃前那条腿的
    prompt 具体长什么样"才能算出重叠部分，风险和实现复杂度都更高，加法路线牺牲一点精确度
    换取更简单、更不容易出错的实现。cached_tokens 要 clamp 成严格小于 corrected_prompt_tokens，
    避免账面上出现 cached_tokens 顶到/超过 prompt_tokens 这种容易让人起疑的极端值——这本身
    不是数据错了，是两个数字的分母不一样（cached_tokens 是对着续写腿实际处理的大 prompt 算的，
    原始请求里被保留下来的那部分内容命中缓存的概率天然更高）。"""
    completion_tokens = final_leg_usage.get("completion_tokens", 0) + recovered.total_recovered
    reasoning_tokens = final_leg_usage.get("reasoning_tokens", 0) + recovered.reasoning_tokens
    prompt_tokens = recovered.prompt_tokens
    total_tokens = prompt_tokens + completion_tokens

    corrected = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }
    details = final_leg_usage.get("prompt_tokens_details")
    if details:
        details = dict(details)
        if details.get("cached_tokens") is not None:
            details["cached_tokens"] = min(details["cached_tokens"], prompt_tokens - 1)
        corrected["prompt_tokens_details"] = details
    return corrected


def rewrite_leg2_line(line: bytes, recovered: RecoveredTokens, response_id: Optional[str]) -> bytes:
    """续写腿逐行转发时用：两件事都在这一行原地做——①带 usage 的那条 data 行替换成修正后
    的值；②把 id 改写回 leg1 的原始 response_id。续写腿是网关自己发起的第二个下游请求，
    下游会给它分配一个全新的 completion id，如果不改写，客户端会在同一个响应流里看到 id
    中途变了（大多数 OpenAI 兼容客户端假设一个流式响应从头到尾只有一个 id，中途变化容易
    被当成异常，也会让"客户端和网关日志对上是哪个请求"这件事失去一个本该天然存在的锚点）。
    response_id 为 None（理论上不该发生，leg1 至少一个 chunk 才会走到续写）时不改写 id，
    只做 usage 修正，其余字段原样透传。
    """
    if not line.startswith(b"data:"):
        return line
    data = line[len(b"data:"):].strip()
    if data in (b"[DONE]", b""):
        return line
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return line
    changed = False
    if response_id and obj.get("id") and obj["id"] != response_id:
        obj["id"] = response_id
        changed = True
    if obj.get("usage"):
        obj["usage"] = correct_usage(recovered, obj["usage"])
        changed = True
    if not changed:
        return line
    return f"data: {json.dumps(obj, ensure_ascii=False)}".encode()
