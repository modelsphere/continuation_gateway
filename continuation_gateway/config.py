import os

DOWNSTREAM_URL = os.environ.get("DOWNSTREAM_URL", "").rstrip("/")
TOKENIZE_URL = os.environ.get("TOKENIZE_URL", DOWNSTREAM_URL).rstrip("/")
PORT = int(os.environ.get("PORT", "8000"))

# 续写逻辑是 Kimi-K3 专属行为（前缀重建按它的 XTML 格式来，换个模型格式就不对了），只对
# 这里列出的 model 名字触发，其余 model 原样透传。不设置时默认空集合——即什么都不触发，
# 逼着部署时显式声明，避免误伤非 Kimi-K3 模型。
CONTINUATION_MODELS = {
    m.strip() for m in os.environ.get("CONTINUATION_MODELS", "").split(",") if m.strip()
}

# 第一个 chunk 等多久都不归这一层管——迟迟没有第一个 chunk 不算"卡住"，是不是要重试是上层
# 的事，不属于续写范畴（没有已经吐给客户端的内容，没什么好救的）。idle timeout 只在收到过
# 至少一个 chunk 之后才生效，见 server.py 的 relay()/relay_leg2()。
STALL_IDLE_TIMEOUT_SECONDS = float(os.environ.get("STALL_IDLE_TIMEOUT_SECONDS", "20"))
CONNECT_TIMEOUT_SECONDS = float(os.environ.get("CONNECT_TIMEOUT_SECONDS", "30"))

# request body 超过这个大小就不进入续写逻辑（只当普通请求透传，卡住/断了也不救）——续写要
# 把 payload 解析成 dict 后整个拿着直到这条流结束（不像 raw_body 用完就能扔），大 body（比如
# 内嵌了图片/视频的多模态请求）意味着这份 dict 要在内存里多待很久；而且真续写时 messages 数组
# 要原样再发两次（一次 /v1/tokenize 一次续写请求本身），大 body 会被重复序列化/传输。默认
# 10MB：上游请求体上限是 100MB（为了兼容视频/图片放开的，原来是 10MB），这里按纯文本 agent
# 场景的实际体量给了充足余量（实测过的真实 Kimi-K3 请求最大也就 100KB 量级），同时明确排除
# 大概率带了图片/视频的大请求，避免续写机制本身在并发场景下变成内存压力的主要来源。以 MB
# 为单位配置（而不是 bytes），部署时设个 10、20 这种直观的数就行，不用心算/背一长串 0。
MAX_CONTINUATION_BODY_MB = int(os.environ.get("MAX_CONTINUATION_BODY_MB", "10"))
MAX_CONTINUATION_BODY_BYTES = MAX_CONTINUATION_BODY_MB * 1024 * 1024
