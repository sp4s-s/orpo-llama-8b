## ✅ Build and Run
```docker
docker build -t orpo-gpu-runner .
docker run --rm --gpus all --env-file .env orpo-gpu-runner

```












Old
# Build the Docker image
docker build -t cuda-env .

# Run the container with .env mounted
docker run --gpus all --env-file env.env -v $(pwd):/app cuda-env


