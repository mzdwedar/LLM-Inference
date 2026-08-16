# Outputs land next to this script (llama-nim/, manifest_out/) - both gitignored.
cd "$(dirname "$0")"

# 1. Pull the amd64 layers into an OCI folder locally
skopeo copy \
  --override-arch amd64 \
  --override-os linux \
  docker://nvcr.io/nim/meta/llama-3.1-8b-instruct:latest \
  oci:llama-nim:latest

# 2. Find and extract the model_manifest.yaml file
mkdir -p manifest_out
find llama-nim/blobs/sha256/ -type f -exec tar -xf {} -C manifest_out "opt/nim/etc/default/model_manifest.yaml" 2>/dev/null \;

# 3. Read all profiles
cat manifest_out/opt/nim/etc/default/model_manifest.yaml