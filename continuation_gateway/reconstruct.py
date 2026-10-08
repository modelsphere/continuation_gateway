"""崩溃前那条腿的 assistant partial 前缀构造。不同模型家族拼 partial assistant 消息用的
控制 token 不一样（Kimi-K3 是自成一套的 XTML 方案，Kimi-K2 系走的是经典 <think>...</think>
sentinel token 方案），按部署所服务的真实模型（config.CONTINUATION_MODEL，不是请求里的
model 字段）分派到各自的构造函数，见 _BUILDERS——没有专门适配过的模型落到
_build_prefix_think_tag（<think>...</think> 格式）：这是目前已知模型里除 Kimi-K3 外
的通用写法，但不保证覆盖所有未来接入的模型，真遇到别的格式要在这里加一个专门的 builder，
不要指望默认格式蒙对。

当前只覆盖自由文本、且崩溃时还没出现过 tool_call chunk 的情况，所以运行时只需要判断
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


def classify_case(tool_calls_seen: bool, reasoning: str, content: str) -> str:
    """崩溃时具体处于哪种情况，日志里的 case 字段用这个字符串。接的是原始信号
    （tool_calls_seen、reasoning/content 是否非空），不是预先算好的字符串——以后要加新
    场景（比如 v2 真的支持 tool_call 续写）时，只需要在这个函数里加新的判断分支/形参，
    调用点大概率不用跟着改，日志格式也不用跟着重新设计。判断顺序按"证据强度"从高到低
    排列，不是随便哪个信号先判都行：

    - "tool-call-seen"：已经出现过 tool_call chunk，最强信号，优先判。这种情况
      总是被 SKIPPED（见 server.py should_intervene 附近的排除逻辑，不会真的走到
      attempt_continuation()），但日志依然如实单独标注这一档，不能因为"反正会被排除"
      就随便套用下面几个不准确的标签——tool_calls_seen=True 时 reasoning/content 通常
      也是空的，如果只按它们是否非空判断会被误判成 "thinking-partial" 甚至 "empty"，
      看起来像是"还在思考"或"什么都没看到"，掩盖了真实情况其实是"已经开始调工具"，这是
      这个函数早期版本（只接 content 一个参数）的真实问题。
    - "content-done"：没有 tool_call，content 非空——真的看到过 content chunk，
      thinking 已经结束，续 content 部分。
    - "thinking-partial"：没有 tool_call、content 为空，但 reasoning 非空——真的看到过
      reasoning chunk，保守判定崩溃时可能还在 thinking 阶段（未必真的没说完，只是网关
      看不到区分信号，见 build_prefix() 里两个 builder 的说明）。这一档必须以"真的见过
      reasoning"为前提，不能只靠"content 是空的"就反推——这个函数早期版本就是这么错的：
      content 为空时不管 reasoning 是不是也是空的，一律叫 thinking-partial，把"确实在
      thinking"和"什么实质内容都没见过"混成了同一个标签。
    - "empty"：tool_call/reasoning/content 都没见到过（比如崩溃前只吐了一个空白的角色
      声明 chunk）。这一档精确对应 `guarded()` 里 `SKIPPED reason=no_recoverable_content`
      那个分支的判断条件（`not (state.reasoning or state.content)` 且未见过
      tool_call）——这种情况没有任何可恢复内容，不会真的走到 attempt_continuation()，
      跟 "content-done"/"thinking-partial" 明确要求"确实见过对应内容"的语义是互斥的，
      不该被套用成其中任何一个。

    跟 build_prefix()/needs_thinking_disabled() 用的是同一条 content 判断依据，这里单独
    抽出来是为了让日志打印和实际选用的前缀构造分支共用同一份逻辑，避免各算一遍悄悄不一致。

    以后 v2 如果真的支持 tool_call 续写，大概率要在 "tool-call-seen" 这一档基础上再细分
    （比如"tool-call-partial"：已出现 tool_call chunk 但还没等到它完整闭合；
    "tool-call-done"：所有已知 tool_call 都已闭合，只是还没等到 finish_reason），需要
    StreamState 追踪比现在的 tool_calls_seen 更细的状态才能区分——这个函数的职责不变，
    到时候加一个新形参、加一个 elif 分支就行，日志字符串本身、调用点的传参习惯都不用改。
    """
    if tool_calls_seen:
        return "tool-call-seen"
    if content:
        return "content-done"
    if reasoning:
        return "thinking-partial"
    return "empty"
