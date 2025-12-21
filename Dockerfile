FROM python:3.11-slim

# 设置工作目录
WORKDIR /home/langchain/workspace

# 设置环境变量
ENV PYTHONPATH=/home/langchain/workspace
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y \
    sudo \
    byobu \
    vim \
    wget \
    gcc \
    g++ \
    curl \
    iputils-ping \
    net-tools \
    iproute2 \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r langchain \
 && useradd -m -r -u 1000 -g langchain -s /bin/bash langchain \
 && echo "langchain:1" | chpasswd

RUN usermod -aG sudo langchain \
 && echo 'langchain ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/langchain \
 && chmod 0440 /etc/sudoers.d/langchain

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /home/langchain/workspace/ && \
    chown -R langchain:langchain /home/langchain/

COPY . .

COPY start.sh /home/langchain/workspace/start.sh

RUN mkdir -p /home/langchain/.local/share/

RUN cp -r file/node/ /usr/local/
RUN cp -r file/.node_modules/ /home/langchain/
RUN cp -r file/pnpm/ /home/langchain/.local/share/
RUN cp -r file/.bashrc /home/langchain/
RUN cp -r file/.oh-my-bash/ /home/langchain/
RUN cp -r file/.osh-update /home/langchain/

RUN rm -r file/

RUN chown -R langchain:langchain /home/langchain/workspace && \
    chmod +x /home/langchain/workspace/start.sh

USER langchain

RUN git clone https://github.com/langchain-ai/agent-chat-ui.git

# 暴露端口
EXPOSE 2024 8000

CMD ["/bin/bash", "/home/langchain/workspace/start.sh"]
