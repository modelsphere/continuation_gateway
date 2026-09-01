"""崩溃前那条腿的 assistant partial 前缀构造。不同模型家族拼 partial assistant 消息用的
控制 token 不一样（Kimi-K3 是自成一套的 XTML 方案，Kimi-K2 系走的是经典 <think>...</think>
sentinel token 方案），按 model 分派到各自的构造函数，见 _BUILDERS——没有专门适配过的 model
落到 _build_prefix_think_tag（<think>...</think> 格式）：这是目前已知模型里除 Kimi-K3 外
的通用写法，但不保证覆盖所有未来接入的模型，真遇到别的格式要在这里加一个专门的 builder，
不要指望默认格式蒙对。

v1 范围只覆盖自由文本、且崩溃时还没出现过 tool_call chunk 的情况，所以运行时只需要判断
content 是否非空：非空说明 thinking 阶段已经结束（content-done），为空则保守按"可能还在
thinking"处理（thinking-partial）——这条对所有 builder 都成立，不是 K3 专属。tool_call
场景、以及 response_format 是 json_object/json_schema 的结构化输出场景，都在 server.py
里更早的地方被排除掉了（前者是崩溃时才发现，后者从原始请求就能判断），走不到这里。
"""

XTML_OPEN = "<|open|>"
XTML_CLOSE = "<|close|>"
XTML_SEP = "<|sep|>"


def _build_prefix_kimi_k3(reasoning: str, content: str) -> str:
    """Kimi-K3 原生的 XTML 控制 token 拼接（<|open|>think<|sep|>...<|close|>think<|sep|>
    <|open|>response<|sep|>...）

    content 非空：thinking 已经闭合，续 content。content 为空：不确定 thinking 是不是刚好
    说完，保守起见不闭合 think 标签，让模型自己决定要不要闭合再转入 response——这样即使
    thinking 其实已经说完，模型接着吐一个闭合标签也是自然的续写；反过来如果这里强行闭合但
    thinking 其实没说完，会截断模型的推理，是不可逆的错误，所以两种不确定性里选风险更小的。
    """
    if content:
        return (
            f"{XTML_OPEN}think{XTML_SEP}{reasoning}{XTML_CLOSE}think{XTML_SEP}"
            f"{XTML_OPEN}response{XTML_SEP}{content}"
        )
    return f"{XTML_OPEN}think{XTML_SEP}{reasoning}"


def _build_prefix_think_tag(reasoning: str, content: str) -> str:
    """默认格式：字面 <think>...</think> 标签包思考内容，content 直接跟在闭合标签后面——
    Kimi-K2 系（K2.5/K2.6 等）用的是这套，不是 K3 的 XTML。"content 是否非空决定要不要闭合
    think 标签"这条保守策略跟 _build_prefix_kimi_k3 一致，理由见那边的说明。
    """
    if content:
        return f"<think>{reasoning}</think>{content}"
    return f"<think>{reasoning}"


_BUILDERS = {
    "kimi-k3": _build_prefix_kimi_k3,
}


def build_prefix(model: str, reasoning: str, content: str) -> str:
    builder = _BUILDERS.get((model or "").lower(), _build_prefix_think_tag)
    return builder(reasoning, content)


def needs_thinking_disabled(content: str) -> bool:
    """对应 continuation_test.py 里的 DISABLE_THINKING：content 已经在生成，说明 thinking
    这一步该关了；content 还没开始，说明还得让模型自己接着想，不能关。这条跟用哪套
    build_prefix 无关，不用按 model 分派。"""
    return bool(content)
