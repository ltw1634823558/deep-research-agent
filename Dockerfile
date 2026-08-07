FROM python:3.12-slim

# 不写 .pyc、日志不缓冲（容器里 stdout 才能实时被采集）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 以非 root 运行：容器逃逸后不直接拿到 root；同时 chown 数据目录，
# 否则 SQLite（memory.db）与 Chroma 落盘会因权限失败。
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查走 /health，编排层可据此自动重启僵死实例
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
