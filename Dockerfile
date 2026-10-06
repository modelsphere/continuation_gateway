# 基础镜像是 build arg，访问不了 Docker Hub 时可以换成自己的镜像仓库。
#
# 公开构建（全部走 Docker Hub，不需要额外参数）：
#   docker build -t continuation-gateway:dev .
#
# 网络受限时，换基础镜像并指定 PyPI 镜像：
#   docker build -t continuation-gateway:dev \
#       --build-arg BUILDER_IMAGE=<registry>/python:3.12-slim \
#       --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .

ARG BUILDER_IMAGE=python:3.12-slim
FROM ${BUILDER_IMAGE}

# 可选的 PyPI 镜像。留空（默认）= 直连 PyPI，也不会覆盖自定义基础镜像里已经配好的 pip 配置。
# pip 会自动读取同名环境变量。
ARG PIP_INDEX_URL=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 必需的运行时环境变量（构建时不需要，容器启动时必须提供，否则 server.py 会直接退出）：
#   DOWNSTREAM_URL       下游路由网关地址，例如 http://router.example.com:8050
#   CONTINUATION_MODELS  逗号分隔的 model 名字，只有列出的 model 才会触发续写，
#                        不设置就默认空集合、什么都不触发
# 可选（默认值见 continuation_gateway/config.py）：
#   PORT（默认 8000）、CONTINUATION_URL（续写请求单独打去另一个地址，不设置就跟
#   DOWNSTREAM_URL 一样）、CONTINUATION_ENABLED（默认 true；设成 false 只监测 leg1、
#   不真的发续写请求）、BUFFER_TOOL_CALLS（默认 false；true 时 tool_call chunk 暂存到
#   完整后才转发，崩溃时丢弃暂存内容并按 tool_call 出现之前的状态续写）、
#   STALL_IDLE_TIMEOUT_SECONDS、CONNECT_TIMEOUT_SECONDS、DOWNSTREAM_CONNECTION_LIMIT、
#   MAX_CONTINUATION_BODY_MB、MAX_REQUEST_BODY_MB、CJK_CHARS_PER_TOKEN、
#   OTHER_CHARS_PER_TOKEN、GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS。
# GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS 默认不设置 = 收到 SIGTERM 后无限等在途请求跑完，兜底
# 上限由部署平台的优雅终止超时负责（例如 Kubernetes 的 terminationGracePeriodSeconds）。
CMD ["python3", "-m", "continuation_gateway.server"]
