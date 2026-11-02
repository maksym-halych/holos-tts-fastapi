#!/usr/bin/env bash
# Boots a built image with no network access and synthesizes real speech. A
# successful `docker build` never loads the model, so this is what proves the
# baked-in checkpoint matches the installed holos-tts architecture.
#
# Usage: scripts/smoke_test_image.sh IMAGE
set -euo pipefail

image="${1:?usage: $0 IMAGE}"
name="holos-smoke-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

docker run -d --name "$name" --network none "$image" >/dev/null

health_check="
import json, urllib.request
assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['status'] == 'ok'
"

ready=false
for _ in $(seq 1 150); do
  if [ "$(docker inspect -f '{{.State.Running}}' "$name")" != "true" ]; then
    echo "Container exited during startup:" >&2
    docker logs "$name" >&2
    exit 1
  fi
  if docker exec "$name" python -c "$health_check" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 2
done

if [ "$ready" != "true" ]; then
  echo "Server did not become ready in time:" >&2
  docker logs "$name" >&2
  exit 1
fi

synthesis_check="
import json, urllib.request
for fmt in ('mp3', 'wav', 'opus', 'flac', 'aac', 'pcm'):
    body = json.dumps({'input': 'Привіт світе!', 'voice': 'Speaker_0', 'response_format': fmt}).encode()
    request = urllib.request.Request('http://127.0.0.1:8000/v1/audio/speech', body, {'Content-Type': 'application/json'})
    size = len(urllib.request.urlopen(request).read())
    assert size > 1000, f'{fmt}: only {size} bytes'
    print(f'{fmt}: {size} bytes')
"

if ! docker exec -e PYTHONIOENCODING=utf-8 "$name" python -c "$synthesis_check"; then
  echo "Synthesis failed:" >&2
  docker logs "$name" >&2
  exit 1
fi

echo "Smoke test passed: $image"
