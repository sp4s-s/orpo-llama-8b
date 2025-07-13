FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

RUN pip install uv

# Set up envs for CUDA and Python
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .

RUN uv pip install --system -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Automatically log in to HF and wandb (token must be passed at runtime)
# Using shell wrapper to run commands before Python entry
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the entrypoint
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default CMD (can be overridden)
CMD ["python", "src/lama_orpo/main.py"]
