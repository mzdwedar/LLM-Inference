# Build context is this directory (holds the Dockerfile).
cd "$(dirname "$0")"

DOCKER_USER="mzdwedar"
IMAGE_NAME="triton-trtllm-light"
TAG="25.01"

# Build for x86_64 architecture using emulation
docker build --platform linux/amd64 -t $DOCKER_USER/$IMAGE_NAME:$TAG .

# Push to Docker Hub
docker push $DOCKER_USER/$IMAGE_NAME:$TAG