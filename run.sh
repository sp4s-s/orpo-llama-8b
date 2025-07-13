#!/bin/bash

# Login again to ensure credentials in runtime shell
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_API_KEY

# Parallel task: upload to HF
python upload_to_hf.py &

# Add other parallel jobs if needed
# python other_script.py &

# Wait for all background jobs to finish
wait
