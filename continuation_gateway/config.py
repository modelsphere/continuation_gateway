import os

DOWNSTREAM_URL = os.environ.get("DOWNSTREAM_URL", "").rstrip("/")
PORT = int(os.environ.get("PORT", "8000"))

# 续写请求（leg2）打去哪个地址——不设置就跟 DOWNSTREAM_URL 一样，只有明确要把续写流量导到
# 另一个集群/服务（比如验证续写机制不占用主链路容量、或者续写用一个规格不同的备用集群）时
# 才需要单独配置。原始请求（leg1，不管最终是走纯转发还是进了续写覆盖）永远打 DOWNSTREAM_URL，
# 只有真的判定要救、发出去的那条续写请求走这个。
CONTINUATION_URL = (os.environ.get("CONTINUATION_URL", "").rstrip("/") or DOWNSTREAM_URL)

# 前缀重建按 model 分派（见 reconstruct.py），只对这里列出的 model 名字触发续写，其余 model
# 原样透传。不设置时默认空集合——即什么都不触发，逼着部署时显式声明。大小写不敏感（存小写，
# 匹配时也把请求里的 model 转小写），避免客户端传的大小写跟这里配置的不一致导致漏判。
CONTINUATION_MODELS = {
    m.strip().lower() for m in os.environ.get("CONTINUATION_MODELS", "").split(",") if m.strip()
}

# 关掉之后网关仍然完整做 leg1 的监测（idle timeout/断连检测、TRIGGERED 判定和留痕都不受
# 影响），只是不真的发第二条腿——用来在不给下游集群增加真实续写请求负载的前提下，验证"这条
# 监测链路本身有没有打通"（能不能正确识别出该救的场景、日志/告警链路走不走得通）。默认打开，
# 不设置这个变量时行为和只有 DOWNSTREAM_URL 一个环境变量的版本完全一致。
CONTINUATION_ENABLED = os.environ.get("CONTINUATION_ENABLED", "true").strip().lower() not in (
    "false", "0", "no", "off")

# tool_call 暂存策略：打开后，leg1 上一旦出现带 tool_calls 的 chunk，就把这一个 chunk 和之后
# 的所有 chunk 暂存在网关内存里不转发，等到 tool_call 全部吐完（收到 finish_reason，或流已经
# 以 [DONE] 收尾）再一次性放给客户端。目的是让"tool_call 已经流给客户端"这种没法续写的状态
# 不会发生——continue_final_message 的语义下 tool_calls 代表这一轮已经说完，客户端拿到
# 一半 tool_call 之后 SGLang 没法接着往下续。暂存期间如果 leg1 卡住/断连，暂存的 tool_call
# 整批丢弃，state 里的 reasoning/content 正好停在第一个 tool_call chunk 之前，按普通的
# thinking-partial/content-done 续写；续写腿之后重新生成（可能又是一个 tool_call）。
# 代价：tool_call 的参数不再逐 chunk 流式到达客户端，而是在 tool_call 完成后一次性到达，
# 且暂存内容在网关内存里停留到 tool_call 结束。默认关闭，此时行为跟没有这个功能时完全一致。
#
# CONTINUATION_ENABLED 是这个开关生效的大前提（见 server.py _forward_leg1_chunk()）：
# CONTINUATION_ENABLED=false 时即使这里是 true 也不会真的暂存——暂存只是为了保住续写的
# 可行性，续写这个动作本身都被关掉了，暂存除了让 tool_call 参数延迟到达之外没有任何用处。
BUFFER_TOOL_CALLS = os.environ.get("BUFFER_TOOL_CALLS", "false").strip().lower() in (
    "true", "1", "yes", "on")

# 第一个 chunk 等多久都不归这一层管——迟迟没有第一个 chunk 不算"卡住"，是不是要重试是上层
# 的事，不属于续写范畴（没有已经吐给客户端的内容，没什么好救的）。idle timeout 只在收到过
# 至少一个 chunk 之后才生效，见 server.py 的 relay()/relay_leg2()。
STALL_IDLE_TIMEOUT_SECONDS = float(os.environ.get("STALL_IDLE_TIMEOUT_SECONDS", "20"))
CONNECT_TIMEOUT_SECONDS = float(os.environ.get("CONNECT_TIMEOUT_SECONDS", "30"))

# 网关到下游（DOWNSTREAM_URL）的所有出向连接共用同一个连接池（server.py on_startup() 建的
# 那个 ClientSession）——不显式设置的话用的是 aiohttp TCPConnector 的库默认值 100。这个上限
# 管的不只是"正在建连"那一下，是"从跟连接器申请连接、到这次请求彻底用完释放"整个区间，
# 包括发完请求头之后老老实实等/读下游流式响应的那一大段时间都占着名额。这个网关的流量形态
# 是每个连接要占用一整条流式响应的时长（可能几秒到几分钟），不是典型 REST API 毫秒级用完
# 就还，100 在真实并发规模下明显偏低——超过这个数的请求不会失败，只是在网关内部排队等空闲
# 槽位，这段排队延迟从客户端侧看跟"下游真的慢"长得一样，容易被误判成下游容量不够，实际是
# 网关自己的连接池在排队。默认给到 500，比库默认值宽松很多；实际部署按目标并发量再调，
# 建议明显高于预期峰值并发，留出余量。
DOWNSTREAM_CONNECTION_LIMIT = int(os.environ.get("DOWNSTREAM_CONNECTION_LIMIT", "500"))

# request body 超过这个大小就不进入续写逻辑（只当普通请求透传，卡住/断了也不救）——续写要
# 把 payload 解析成 dict 后整个拿着直到这条流结束（不像 raw_body 用完就能扔），大 body（比如
# 内嵌了图片/视频的多模态请求）意味着这份 dict 要在内存里多待很久。默认 10MB：上游请求体
# 上限是 100MB（为了兼容视频/图片放开的，原来是 10MB），这里按纯文本 agent 场景的实际体量
# 给了充足余量（典型纯文本 agent 请求最大也就 100KB 量级），同时明确排除大概率带了
# 图片/视频的大请求，避免续写机制本身在并发场景下变成内存压力的主要来源。以 MB 为单位配置
# （而不是 bytes），部署时设个 10、20 这种直观的数就行，不用心算/背一长串 0。
MAX_CONTINUATION_BODY_MB = int(os.environ.get("MAX_CONTINUATION_BODY_MB", "10"))
MAX_CONTINUATION_BODY_BYTES = MAX_CONTINUATION_BODY_MB * 1024 * 1024

# aiohttp 的 web.Application 默认 client_max_size 只有 1MB，比 MAX_CONTINUATION_BODY_MB
# 本身还小——如果不显式覆盖，任何超过 1MB 的请求体（不管是不是续写模型、不管要不要触发续写）
# 都会在业务代码跑起来之前就被 aiohttp 框架直接拒成 413，MAX_CONTINUATION_BODY_MB 这个"超限
# 就纯透传"的设计会形同虚设。默认给到跟上游请求体上限（100MB，为兼容图片/视频放开的）一致，
# 保证网关这一层不会比它上游更严格地拒绝合法请求。
MAX_REQUEST_BODY_MB = int(os.environ.get("MAX_REQUEST_BODY_MB", "100"))
MAX_REQUEST_BODY_BYTES = MAX_REQUEST_BODY_MB * 1024 * 1024

# usage 修正 / max_tokens 扣减要用到的 token 数，不再靠 /v1/tokenize 现测（避免对下游
# SGLang 服务引入额外依赖），改成按字符数估算，见 usage.py
# 的 estimate_tokens()。CJK（中/日/韩）字符和其它字符分开算，因为两者在大多数 BPE 分词器里
# "一个字符占多少 token"的密度差别很大；这两个比例是通用经验值，不是针对某个具体分词器
# 校准过的精确值，如果后续拿到真实样本可以调整这两个默认值。
CJK_CHARS_PER_TOKEN = float(os.environ.get("CJK_CHARS_PER_TOKEN", "1.6"))
OTHER_CHARS_PER_TOKEN = float(os.environ.get("OTHER_CHARS_PER_TOKEN", "4.0"))

# 收到 SIGTERM（k8s 删除/滚动更新 pod 时先发这个）之后，aiohttp 的 web.run_app() 会自动停止
# 监听新连接，然后等现有请求跑完再退出——"最多等多久"这个上限就是这个值，传给
# web.run_app(shutdown_timeout=...)。当前版本的设计意图是老老实实等所有在途请求自然结束，
# 不主动截断（真正的兜底上限由部署平台的优雅终止超时负责，例如 Kubernetes 的
# terminationGracePeriodSeconds，到点了会直接 SIGKILL，这里不用也不该再重复设一个更短的
# 人为上限跟它打架；部署时应把它配得足够长）。
# 空字符串/不设置 = 无限等待——传 None 给 aiohttp 会让它用 async_timeout.timeout(None)，
# 真正意义上的不设超时，不是"设一个很大的数"那种近似。
# 后续如果改成"关闭时把还在处理的请求转发/推给续写服务，而不是在本地死等"，这里大概率会
# 需要换成一个有限值，把这个开关留着方便到时候切换，不用再改调用方代码。
_shutdown_timeout_raw = os.environ.get("GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS", "").strip()
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = float(_shutdown_timeout_raw) if _shutdown_timeout_raw else None
