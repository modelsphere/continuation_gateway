"""崩溃前那条腿的 assistant partial 前缀构造，用 Kimi-K3 原生的 XTML 控制 token 拼接
（<|open|>think<|sep|>...<|close|>think<|sep|><|open|>response<|sep|>...），不走 OpenAI
标准 message.tool_calls 那条结构化字段的路——那条路要让 SGLang 原生支持需要把任意客户端
拼出来的结构化字段转回模型训练时见过的续写状态，改造成本过高，这里不实现。

v1 范围只覆盖自由文本、且崩溃时还没出现过 tool_call chunk 的情况，所以运行时只需要判断
content 是否非空：非空说明 thinking 阶段已经结束（content-done），为空则保守按"可能还在
thinking"处理（thinking-partial）——见 build_prefix 里的说明。tool_call 场景、以及
response_format 是 json_object/json_schema 的结构化输出场景，都在 server.py 里更早的
地方被排除掉了（前者是崩溃时才发现，后者从原始请求就能判断），走不到这里——json-content-done
这个"结构上等价于 content-done、只是 content 是被截断的 JSON"的场景理论上可以复用这同一套
构造，但 v1 干脆不覆盖，不需要在这个函数里做任何区分。
"""

XTML_OPEN = "<|open|>"
XTML_CLOSE = "<|close|>"
XTML_SEP = "<|sep|>"


def build_prefix(reasoning: str, content: str) -> str:
    """content 非空：thinking 已经闭合，续 content。content 为空：不确定 thinking 是不是刚好
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


def needs_thinking_disabled(content: str) -> bool:
    """对应 continuation_test.py 里的 DISABLE_THINKING：content 已经在生成，说明 thinking
    这一步该关了；content 还没开始，说明还得让模型自己接着想，不能关。"""
    return bool(content)
