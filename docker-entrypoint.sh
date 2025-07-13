#!/bin/bash

# Hugging Face login if token exists
if [ -n "$HF_TOKEN" ]; then
  echo "$HF_TOKEN" | huggingface-cli login --token --stdin > /dev/null
fi

# Weights & Biases login if API key exists
if [ -n "$WANDB_API_KEY" ]; then
  wandb login "$WANDB_API_KEY" > /dev/null
fi

# Run actual container CMD
exec "$@"
