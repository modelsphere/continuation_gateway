import os

DOWNSTREAM_URL = os.environ.get("DOWNSTREAM_URL", "").rstrip("/")
PORT = int(os.environ.get("PORT", "8000"))

# 前缀重建按 model 分派（见 reconstruct.py），只对这里列出的 model 名字触发续写，其余 model
# 原样透传。不设置时默认空集合——即什么都不触发，逼着部署时显式声明。大小写不敏感（存小写，
# 匹配时也把请求里的 model 转小写），避免客户端传的大小写跟这里配置的不一致导致漏判。
CONTINUATION_MODELS = {
    m.strip().lower() for m in os.environ.get("CONTINUATION_MODELS", "").split(",") if m.strip()
}

# 第一个 chunk 等多久都不归这一层管——迟迟没有第一个 chunk 不算"卡住"，是不是要重试是上层
# 的事，不属于续写范畴（没有已经吐给客户端的内容，没什么好救的）。idle timeout 只在收到过
# 至少一个 chunk 之后才生效，见 server.py 的 relay()/relay_leg2()。
STALL_IDLE_TIMEOUT_SECONDS = float(os.environ.get("STALL_IDLE_TIMEOUT_SECONDS", "20"))
CONNECT_TIMEOUT_SECONDS = float(os.environ.get("CONNECT_TIMEOUT_SECONDS", "30"))

# request body 超过这个大小就不进入续写逻辑（只当普通请求透传，卡住/断了也不救）——续写要
# 把 payload 解析成 dict 后整个拿着直到这条流结束（不像 raw_body 用完就能扔），大 body（比如
# 内嵌了图片/视频的多模态请求）意味着这份 dict 要在内存里多待很久。默认 10MB：上游请求体
# 上限是 100MB（为了兼容视频/图片放开的，原来是 10MB），这里按纯文本 agent 场景的实际体量
# 给了充足余量（实测过的真实 Kimi-K3 请求最大也就 100KB 量级），同时明确排除大概率带了
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

# usage 修正 / max_tokens 扣减要用到的 token 数，不再靠 /v1/tokenize 现测（避免更改下游
# SGLang服务），改成按字符数估算，见 usage.py
# 的 estimate_tokens()。CJK（中/日/韩）字符和其它字符分开算，因为两者在大多数 BPE 分词器里
# "一个字符占多少 token"的密度差别很大；这两个比例是通用经验值，不是针对 Kimi-K3 分词器
# 校准过的精确值，如果后续拿到真实样本可以调整这两个默认值。
CJK_CHARS_PER_TOKEN = float(os.environ.get("CJK_CHARS_PER_TOKEN", "1.6"))
OTHER_CHARS_PER_TOKEN = float(os.environ.get("OTHER_CHARS_PER_TOKEN", "4.0"))
