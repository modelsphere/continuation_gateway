FROM harbor.4pd.io/sagegpt-aio/pk_platform/ubuntu_python:24.04_3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8000

# 必需的运行时环境变量（构建时不需要，容器启动时必须提供，否则 server.py 会直接退出）：
#   DOWNSTREAM_URL       下游路由网关地址，例如 http://172.26.3.82:8050
#   CONTINUATION_MODELS  逗号分隔的 Kimi-K3 model 名字，不设置就默认空集合、什么都不触发
# 可选：PORT（默认 8000）、CONTINUATION_URL（续写请求单独打去另一个地址，不设置就跟
#      DOWNSTREAM_URL 一样）、CONTINUATION_ENABLED（默认 true；设成 false 只监测 leg1、
#      不真的发续写请求）、STALL_IDLE_TIMEOUT_SECONDS、CONNECT_TIMEOUT_SECONDS、
#      DOWNSTREAM_CONNECTION_LIMIT、MAX_CONTINUATION_BODY_MB、MAX_REQUEST_BODY_MB、
#      CJK_CHARS_PER_TOKEN、OTHER_CHARS_PER_TOKEN、GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS
#      （默认值见 config.py）。GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS 默认不设置=收到 SIGTERM
#      后无限等在途请求跑完，兜底上限由部署清单里的 terminationGracePeriodSeconds 负责。
CMD ["python3", "-m", "continuation_gateway.server"]
