FROM harbor-contest.4pd.io/luxinlong02/python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8000

# 必需的运行时环境变量（构建时不需要，容器启动时必须提供，否则 server.py 会直接退出）：
#   DOWNSTREAM_URL       下游路由网关地址，例如 http://172.26.3.82:8050
#   CONTINUATION_MODELS  逗号分隔的 Kimi-K3 model 名字，不设置就默认空集合、什么都不触发
# 可选：TOKENIZE_URL（默认同 DOWNSTREAM_URL）、PORT（默认 8000）、
#      STALL_IDLE_TIMEOUT_SECONDS、CONNECT_TIMEOUT_SECONDS、MAX_CONTINUATION_BODY_BYTES
#      （默认值见 config.py）。
CMD ["python", "-m", "continuation_gateway.server"]
