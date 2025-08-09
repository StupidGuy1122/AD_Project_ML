# 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 拷贝依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 暴露 80 端口
EXPOSE 80

# 启动 FastAPI 应用（固定监听 0.0.0.0:80）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
