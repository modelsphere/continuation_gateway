"""usage/账单修正。

崩溃的那条腿由于连接中断，永远不会上报 usage；续写这条腿自己的 usage 是"这条腿单独生成了
多少"，不包含被崩溃吞掉、又靠续写救回来的那部分内容——如果直接把续写腿的 usage 转发给客户端，
会看到 completion_tokens 异常小、看起来像是内容丢了。这里的公式只做加法，不追求绝对精确：
`corrected_prompt_tokens` 用 /v1/tokenize 的 messages 模式重新测一次原始请求（走跟
/v1/chat/completions 相同的 chat_template 渲染），`corrected_completion/reasoning_tokens`
= 续写腿真实上报的值 + 被救回的 reasoning/content 文本单独测出来的 token 数。

已知阻塞项：Kimi-K3 部署上 /v1/tokenize 的 messages 模式目前返回 500（疑似 SGLang bug，
待确认 traceback），这条链路打通前，measure_recovered_tokens 会抛异常，调用方（server.py 的
attempt_continuation）会捕获并降级——续写照常进行、只是不做 max_tokens 精确扣减和 usage
修正，不能让一个 usage 统计的 bug 挡住给客户端的正文内容恢复。
"""

import json


class RecoveredTokens:
    __slots__ = ("prompt_tokens", "reasoning_tokens", "content_tokens")

    def __init__(self, prompt_tokens: int, reasoning_tokens: int, content_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.reasoning_tokens = reasoning_tokens
        self.content_tokens = content_tokens

    @property
    def total_recovered(self) -> int:
        return self.reasoning_tokens + self.content_tokens


async def _tokenize(session, tokenize_url: str, body: dict) -> int:
    async with session.post(f"{tokenize_url}/v1/tokenize", json=body) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["count"]


async def tokenize_messages(session, tokenize_url: str, original_payload: dict, model: str) -> int:
    body = {"model": model, "messages": original_payload["messages"]}
    for key in ("tools", "tool_choice", "chat_template_kwargs"):
        if key in original_payload:
            body[key] = original_payload[key]
    return await _tokenize(session, tokenize_url, body)


async def tokenize_text(session, tokenize_url: str, text: str, model: str) -> int:
    if not text:
        return 0
    # add_special_tokens=false：Kimi-K3 的 tokenizer 只有在零 kwargs 调用时才走它自己干净的
    # 自定义编码路径（不额外加 BOS/EOS），这正是真实续写把 assistant 前缀拼进 prompt 时的
    # 编码方式；/v1/tokenize 的实现只要传了 add_special_tokens（不管真假）就会改走标准 HF
    # 通用兜底路径，行为不完全等价。传 false 至少能保证不会比真实续写多算出一个 BOS，是能做到
    # 的最小偏差版本。
    body = {"model": model, "prompt": text, "add_special_tokens": False}
    return await _tokenize(session, tokenize_url, body)


async def measure_recovered_tokens(session, tokenize_url: str, original_payload: dict,
                                    reasoning_text: str, content_text: str) -> RecoveredTokens:
    model = original_payload.get("model", "llm")
    prompt_tokens = await tokenize_messages(session, tokenize_url, original_payload, model)
    reasoning_tokens = await tokenize_text(session, tokenize_url, reasoning_text, model)
    content_tokens = await tokenize_text(session, tokenize_url, content_text, model)
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


def rewrite_usage_line(line: bytes, recovered) -> bytes:
    """续写腿逐行转发时用：其余行原样透传，只有带 usage 的那条 data 行原地替换成修正后的值。
    recovered 为 None（tokenize 失败降级）时不做任何修改，原样透传最后一条腿的真实 usage。
    """
    if recovered is None or not line.startswith(b"data:"):
        return line
    data = line[len(b"data:"):].strip()
    if data in (b"[DONE]", b""):
        return line
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return line
    if not obj.get("usage"):
        return line
    obj["usage"] = correct_usage(recovered, obj["usage"])
    return f"data: {json.dumps(obj, ensure_ascii=False)}".encode()
