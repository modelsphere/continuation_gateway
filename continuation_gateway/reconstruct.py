"""崩溃前那条腿的 assistant partial 前缀构造。前缀怎么拼由部署的「引擎 + 真实模型」决定
（config.CONTINUATION_ENGINE / config.CONTINUATION_MODEL，不看请求里的 model 字段），续写腿按
引擎分两种形态。

chat-continue（两个引擎的默认形态）：最后一条 assistant 消息带上前缀，发 continue_final_message。
前缀 builder 按引擎登记在 _PREFIX_BUILDERS 里，没登记的组合落到 _build_prefix_think_tag
（<think>...</think> 格式）：这是多数模型的通用写法，但不保证覆盖所有未来接入的模型，真遇到别的
格式要加一个专门的 builder，不要指望默认格式蒙对。两个引擎分开登记，是因为同一个前缀在两个引擎上
不一定都成立：SGLang 的 reasoning parser 根据 previous_content（最后一条 assistant 消息的
content）里有没有模型自己的思考起止 token 决定续写从哪个状态起步，前缀不用模型自己的 token 时，
续写出的思考会落进 content、收尾标记原样漏出；vLLM 要求最后一条消息的 content 原样出现在渲染
结果里，有的模板会剥掉或重排 assistant content 里的思考段（如 Gemma-4 的 strip_thinking），
在 SGLang 上可用的原生前缀到 vLLM 上可能直接被拒（400）。vLLM 上另有一类组合在 content-done 时
改发 reasoning / content 分开的结构化消息，见 _STRUCTURED_CONTENT_DONE 和 build_assistant_message()。

render-generate（只有 vLLM 有这套接口）：渲染器不认 continue_final_message 的（Kimi-K3），或者
模板会改写最后一条 assistant 消息、chat-continue 必然被拒的（MiniMax-M2.7 的模板会把 content 里的
`<think>R</think>` 拆出去；Gemma-4 的模板会剥掉 `<|channel>...<channel|>`），续写腿改成 render →
tokenize → generate → derender 四步，见 _VLLM_RENDER_GENERATE 和 render_generate.py。这条路径的
前缀不带开头标签（render 出来的 prompt 已经打开了思考通道），默认用 build_render_prefix() 的通用
写法，Kimi-K3 用 build_k3_render_prefix()，Gemma-4 用 build_gemma4_render_prefix()（generation
prompt 不一定打开思考通道，开头按请求判断）。

模型名按 match_key() 匹配：归一化（大小写、空白、下划线）后精确匹配。表里的 key 就是已验证的
完整型号，也是 CONTINUATION_MODEL 应该设置的字符串；已验证模型的量化版、微调版也应该直接
设置成该型号的名字（如 GLM-5.3-FP8 设置成 glm-5.3），没验证过的型号和版本不会被误命中。

当前只覆盖自由文本、且崩溃时还没出现过 tool_call chunk 的情况，所以运行时只需要判断
content 是否非空：非空说明 thinking 阶段已经结束（content-done），为空则保守按"可能还在
thinking"处理（thinking-partial）——这条对所有 builder 都成立，不是 K3 专属。tool_call
场景、以及 response_format 是 json_object/json_schema 的结构化输出场景，都在 server.py
里更早的地方被排除掉了（前者是崩溃时才发现，后者从原始请求就能判断），走不到这里。
"""

import re

XTML_OPEN = "<|open|>"
XTML_CLOSE = "<|close|>"
XTML_SEP = "<|sep|>"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


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
        return f"{THINK_OPEN}{reasoning}{THINK_CLOSE}{content}"
    return f"{THINK_OPEN}{reasoning}"


GEMMA4_CHANNEL_OPEN = "<|channel>"
GEMMA4_CHANNEL_CLOSE = "<channel|>"
GEMMA4_THOUGHT_LABEL = "thought\n"


def _build_prefix_gemma4(reasoning: str, content: str) -> str:
    """Gemma-4 原生的思考通道：`<|channel>thought\\n{reasoning}<channel|>{content}`，与 SGLang
    Gemma4Detector 的 think_start_token / think_start_self_label / think_end_token 一致。
    content 非空时总是写出完整的通道（reasoning 为空就是空通道，模板在不思考时也写空通道）；
    content 为空时通道不闭合，理由见 _build_prefix_kimi_k3。"""
    if content:
        return f"{GEMMA4_CHANNEL_OPEN}{GEMMA4_THOUGHT_LABEL}{reasoning}{GEMMA4_CHANNEL_CLOSE}{content}"
    return f"{GEMMA4_CHANNEL_OPEN}{GEMMA4_THOUGHT_LABEL}{reasoning}"


HY3_THINK_OPEN = "<think:opensource>"
HY3_THINK_CLOSE = "</think:opensource>"


def _build_prefix_hy3(reasoning: str, content: str) -> str:
    """Hy3 的思考 token 带 `:opensource` 后缀（词表里的 added token，SGLang HunyuanDetector 从
    词表取出带后缀的 token 作为思考起止）。字面 `</think>` 不会让解析器切出思考状态：
    reasoning_effort 不是 none 时解析器从思考状态起步，content-done 用 `</think>` 闭合时续写的
    正文会整段落进 reasoning。content 非空时总是写出完整的 think 块（reasoning 为空就是空块，
    与模板 no_think 时的 generation prompt 一致）；content 为空时不闭合，理由见 _build_prefix_kimi_k3。"""
    if content:
        return f"{HY3_THINK_OPEN}{reasoning}{HY3_THINK_CLOSE}{content}"
    return f"{HY3_THINK_OPEN}{reasoning}"


MISTRAL_THINK_OPEN = "[THINK]"
MISTRAL_THINK_CLOSE = "[/THINK]"


def _build_prefix_mistral(reasoning: str, content: str) -> str:
    """Mistral tekken 新版的思考 special token `[THINK]` / `[/THINK]`，与 SGLang MistralDetector 和
    模型自带模板渲染 reasoning_content 的方式一致。content 非空、reasoning 为空时只写 content：
    模板在不思考时不写空的 think 块。content 为空时不闭合，理由见 _build_prefix_kimi_k3。

    前缀里的 `[THINK]` 要被编码回 special token 才有意义：引擎需要用 HF tokenizer（tokenizer.json
    里这两个是 special token），mistral_common tokenizer 不保证把字面文本还原成控制 token。"""
    if content:
        if not reasoning:
            return content
        return f"{MISTRAL_THINK_OPEN}{reasoning}{MISTRAL_THINK_CLOSE}{content}"
    return f"{MISTRAL_THINK_OPEN}{reasoning}"


# 引擎 -> 模型型号 -> chat-continue 的前缀 builder，没登记的落到 _build_prefix_think_tag，
# 两个引擎分开登记的原因见模块说明。
_PREFIX_BUILDERS = {
    "sglang": {
        "kimi-k3": _build_prefix_kimi_k3,
        "kimi-k3.1-pro": _build_prefix_kimi_k3,
        "gemma-4-26b-a4b-it": _build_prefix_gemma4,
        "gemma-4-31b-it": _build_prefix_gemma4,
        "hy3": _build_prefix_hy3,
        "mistral-small-4-119b-2603": _build_prefix_mistral,
    },
    "vllm": {},
}

# 已验证过的「引擎 -> 模型型号」组合（README 兼容表里的模型名）。只用于启动时判断
# CONTINUATION_MODEL 在当前 CONTINUATION_ENGINE 上有没有验证过，不参与任何分发；没有专门
# builder 的型号（如 glm-5.3）走默认 <think> 格式，也在这里登记。vLLM 上的 Kimi-K3、
# MiniMax-M2.7 和 Gemma-4 走 render-generate 路径（见 uses_render_generate()），要求 vLLM 开启
# --enable-scale-out。
VERIFIED_MODELS = {
    "sglang": ("kimi-k3", "kimi-k3.1-pro", "qwen3.6-35b-a3b", "qwen3.8-27b", "glm-5.3",
               "nemotron-3.5-lightning-30b-a3b", "mimo-v2.5", "deepseek-v4.1-flash-1017",
               "step-3.5-flash", "qwen3.8-flash-next", "ling-3.0-flash-vl", "nex-n2.5-pro",
               "gemma-4-26b-a4b-it", "gemma-4-31b-it", "hy3", "mistral-small-4-119b-2603",
               "laguna-s-2.1"),
    "vllm": ("qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3-30b-a3b", "qwen3-14b", "qwen3-8b",
             "gemma-4-31b-it", "gemma-4-26b-a4b-it", "glm-5.3", "deepseek-v4-flash-0731-nvfp4",
             "kimi-k3", "minimax-m2.7"),
}

# 关思考开关在 chat_template_kwargs 里的 key：默认 `thinking`，Qwen3 系的模板读的是
# `enable_thinking`（对 `thinking` 无反应），在这里登记。续写下游约定是裸引擎 URL，模板变量
# 只能通过 chat_template_kwargs 传入，所以只有这一种写法。每个模型只发一个 key：有的模型
# 收到不认识的开关会把 reasoning 漏进 content。
_DEFAULT_THINKING_SWITCH_KEY = "thinking"
_THINKING_SWITCH_KEYS = {
    "qwen3.6-35b-a3b": "enable_thinking",
    "qwen3.6-27b": "enable_thinking",
    "qwen3-30b-a3b": "enable_thinking",
    "qwen3-14b": "enable_thinking",
    "qwen3-8b": "enable_thinking",
}

# 引擎 -> 该引擎上 content-done 需要发结构化消息的模型型号。只登记测试出有问题确实需要的
_STRUCTURED_CONTENT_DONE = {
    "vllm": ("qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3-8b"),
}


def _normalize(name: str) -> str:
    return re.sub(r"[\s_]+", "-", (name or "").strip().lower())


def match_key(model: str, keys) -> str:
    """model 归一化后与 keys 里的某个 key 相等则返回该 key，否则返回 None。"""
    name = _normalize(model)
    for key in keys:
        if _normalize(key) == name:
            return key
    return None


# vLLM 上续写腿走 render-generate 的模型型号。只有 vLLM 有这套接口，所以不按引擎分层。前缀默认用
# build_render_prefix() 的通用写法；需要专属处理的模型（Kimi-K3、Gemma-4）在 render_generate.py
# 里分发。Gemma-4 的模板会剥掉 assistant content 里的 `<|channel>...<channel|>`，原生前缀走
# chat-continue 会被拒（400），所以也走这条路径。
_VLLM_RENDER_GENERATE = ("kimi-k3", "minimax-m2.7", "gemma-4-26b-a4b-it", "gemma-4-31b-it")


def uses_render_generate(engine: str, model: str) -> bool:
    return engine == "vllm" and bool(match_key(model, _VLLM_RENDER_GENERATE))


def build_prefix(engine: str, model: str, reasoning: str, content: str) -> str:
    """chat-continue 形态的字符串前缀。"""
    table = _PREFIX_BUILDERS.get(engine, {})
    builder = table.get(match_key(model, table), _build_prefix_think_tag)
    return builder(reasoning, content)


def build_assistant_message(engine: str, model: str, reasoning: str, content: str) -> dict:
    """续写请求末尾那条 partial assistant 消息。

    默认就是 build_prefix() 拼出的字符串前缀。_STRUCTURED_CONTENT_DONE 里登记的「引擎 + 模型」
    且 content 已经开始（content-done）时例外：改成 reasoning / content 分开的结构化消息。
    原因是 continue_final_message 在渲染 chat template 之后，要求最后一条消息的 content 原样
    出现在渲染结果里，否则引擎直接拒绝请求（400）。有的模板（比如 Qwen3.6）会把 content 里的
    `<think>R</think>C` 拆开重排，在 `<think>` 后和 `</think>` 前后补换行并去掉首尾空白，整段
    字符串就不再原样出现；而 reasoning 单独传时，模板按 reasoning_content 渲染思考部分，
    content 只剩正文 C，原样出现，截断位置恰好在 C 末尾。
    thinking-partial（content 为空）没有 `</think>`，模板不会重排，仍然用字符串前缀；结构化
    消息也表达不了「思考还没结束」，模板会把 think 闭合。
    """
    if content and match_key(model, _STRUCTURED_CONTENT_DONE.get(engine, ())):
        return {"role": "assistant", "reasoning": reasoning, "content": content}
    return {"role": "assistant", "content": build_prefix(engine, model, reasoning, content)}


def thinking_disable_kwargs(model: str, content: str) -> dict:
    """续写请求需要合并进 chat_template_kwargs 的「关思考」开关。

    content 已经在生成，说明 thinking 这一步该关了；content 还没开始，说明还得让模型自己
    接着想，不能关，返回空字典。开关的 key 因模型的 chat template 而异，只发一个：
    _THINKING_SWITCH_KEYS 登记了与默认 `thinking` 不同的模型，其它模型（包括 Kimi-K3、没有
    外部开关的 GLM-5.3，以及未验证的模型）用 `thinking`——模板里没有引用的变量会被忽略。"""
    if not content:
        return {}
    key = _THINKING_SWITCH_KEYS.get(match_key(model, _THINKING_SWITCH_KEYS), _DEFAULT_THINKING_SWITCH_KEY)
    return {key: False}


def build_k3_render_prefix(reasoning: str, content: str, thinking_open: bool) -> str:
    """render-generate 路径的 Kimi-K3 前缀：接在 render 出来的 prompt 后面、模型接着 generation
    prompt 会写出来的原始文本。和 _build_prefix_kimi_k3 不同，这里不写开头的
    `<|open|>think<|sep|>`，因为 prompt 末尾已经带了。

    thinking_open 由 prompt 末尾的 token 决定（`<|open|>think<|sep|>` 为真，
    `<|open|>response<|sep|>` 为假），不靠「reasoning 是否为空」推断：
    - 思考通道开着：content 为空时就是裸的 reasoning（不闭合，让模型自己决定何时收尾）；
      content 非空时 reasoning 后面接 think 的闭合和 response 的开头，再接 content。
    - 思考通道已关（prompt 末尾直接是 response 的开头）：只有 content，不能再写任何标签；
      此时不该有 reasoning，有就说明 prompt 和 leg1 的实际输出对不上，抛 ValueError。
    """
    if not thinking_open:
        if reasoning:
            raise ValueError("prompt closes the thinking channel but leg1 produced reasoning")
        return content
    if content:
        return (
            f"{reasoning}{XTML_CLOSE}think{XTML_SEP}"
            f"{XTML_OPEN}response{XTML_SEP}{content}"
        )
    return reasoning


def build_render_prefix(reasoning: str, content: str) -> str:
    """render-generate 路径的通用前缀：generation prompt 末尾已经打开了思考，这里不写开头标签。
    reasoning 为空说明没有思考（或请求关了思考，prompt 末尾是闭合的空思考块），只接 content，
    不能再写闭合标签。思考是开是关由 derender 的 parser 按完整 prompt 判断。"""
    if content:
        return f"{reasoning}{THINK_CLOSE}{content}" if reasoning else content
    return reasoning


def gemma4_channel_prewritten(payload: dict) -> bool:
    """Gemma-4 的 generation prompt 是否已经写了 `<|channel>thought\\n`：最后一条消息是 tool 回复、
    且 chat_template_kwargs.enable_thinking 为真时，模板在同一个 model 回合里接着写，并预先打开思考
    通道；其余情况 generation prompt 只写到 `<|turn>model\\n`（思考打开时通道由模型自己打开），或
    写一个闭合的空通道（思考关闭）。"""
    messages = payload.get("messages") or []
    thinking = (payload.get("chat_template_kwargs") or {}).get("enable_thinking")
    return bool(messages) and messages[-1].get("role") == "tool" and bool(thinking)


def build_gemma4_render_prefix(reasoning: str, content: str, channel_prewritten: bool) -> str:
    """render-generate 路径的 Gemma-4 前缀。Gemma-4 不满足「generation prompt 已经打开思考」这个
    通用前提，开头要按 channel_prewritten（见 gemma4_channel_prewritten()）决定：已预写时不写，
    否则前缀自己带 `<|channel>thought\\n`（和 chat-continue 的 _build_prefix_gemma4 相同）。闭合用
    `<channel|>`，不是 `</think>`。reasoning 为空（思考关闭，prompt 末尾已是闭合的空通道）时只接
    content。思考是开是关由 derender 的 gemma4 parser 从完整 prompt 末尾往回找 `<|channel>` 判断。"""
    head = "" if channel_prewritten else f"{GEMMA4_CHANNEL_OPEN}{GEMMA4_THOUGHT_LABEL}"
    if content:
        return f"{head}{reasoning}{GEMMA4_CHANNEL_CLOSE}{content}" if reasoning else content
    return f"{head}{reasoning}"


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

    跟 build_prefix()/thinking_disable_kwargs() 用的是同一条 content 判断依据，这里单独
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
