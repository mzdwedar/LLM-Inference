# Start from the base Triton image (~7.3 GB compressed)
FROM nvcr.io/nvidia/tritonserver:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install basic OS utilities & Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies safely without upgrading system pip
RUN pip install --no-cache-dir --break-system-packages --ignore-installed setuptools wheel

# Install TensorRT-LLM safely using a constraint file
RUN CURRENT_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null) && \
    echo "torch==$CURRENT_TORCH" > /tmp/torch-constraint.txt && \
    pip install --no-cache-dir --break-system-packages \
        tensorrt_llm \
        --extra-index-url https://pypi.nvidia.com \
        -c /tmp/torch-constraint.txt

# Install Python backend essentials for Triton
RUN pip install --no-cache-dir --break-system-packages \
    transformers \
    tokenizers \
    accelerate \
    sentencepiece \
    pydantic \
    geventhttpclient

WORKDIR /workspace

CMD ["tritonserver", "--model-repository=/opt/tritonserver/model_repository"]