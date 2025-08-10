# ===== 构建阶段 =====
FROM python:3.10-slim AS builder

WORKDIR /app

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc wget curl && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 升级 pip 并安装 PyTorch（CPU 版）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.5.1+cpu \
        torchaudio==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu

# 安装其余依赖
RUN pip install --no-cache-dir -r requirements.txt

# ===== 运行阶段 =====
FROM python:3.10-slim

WORKDIR /app

# 安装基础运行依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl && \
    rm -rf /var/lib/apt/lists/*

# 从 builder 复制 Python 运行环境
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码
COPY . .

# 环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 80

# 启动 FastAPI 应用
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:80", "--workers", "4", "--timeout", "120"]
