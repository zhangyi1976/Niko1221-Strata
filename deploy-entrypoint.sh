#!/bin/sh
# Entry point for the Strata image when it runs as a GPUStack custom inference
# backend. It is the project's docker-entrypoint.sh plus the two things GPUStack
# adds: the model path and the port, passed in as the first two arguments by the
# backend's run_command ({{model_path}} and {{port}} in deploy/backend.yaml).
#
# Model source, auto-detected:
#   - If $1 is a directory that already holds GGUF shards (or a single .gguf
#     file), setup.py reads them from there with --gguf-dir. This is the
#     GPUStack-native path: GPUStack downloaded the model files itself. Only the
#     prepared pack and the ~5 GB MTP draft layer are fetched, into /data.
#   - Otherwise the model is downloaded by Strata on the first start, exactly as
#     the project's own Docker image does, into /data.
#
# Every other setting is the same env var the project's image uses: FAMILY, MODEL,
# CONTEXT, VISION, KV, GPU, GPUS, LAYER_SPLIT, LOW_RAM, HOST, PORT, API_KEY,
# STRATA_DATA, REINSTALL.
set -e
cd /opt/strata || exit 1

# ---- the two values GPUStack passes in ----------------------------------------
PORT_ENV="${PORT:-8080}"
MODEL_PATH="${1:-}"
PORT="${2:-}"

# A placeholder that was never substituted ("{{model_path}}" or "{{port}}") is
# treated as unset, and an empty port falls back to the PORT env var or 8080.
case "$MODEL_PATH" in *"{{"*) MODEL_PATH="" ;; esac
case "$PORT" in *"{{"*) PORT="" ;; esac
PORT="${PORT:-$PORT_ENV}"

# ---- where the GGUFs come from ------------------------------------------------
# --gguf-dir expects the folder that holds every shard. If $1 is a single .gguf
# file, its parent folder is that folder.
GGUF_DIR=""
if [ -n "$MODEL_PATH" ]; then
  if [ -d "$MODEL_PATH" ]; then
    if ls "$MODEL_PATH"/*.gguf >/dev/null 2>&1; then
      GGUF_DIR="$MODEL_PATH"
    else
      # The directory exists but the shards are not all there yet. GPUStack
      # downloads the model files around the time it schedules the container, so
      # wait for the first shard to appear before deciding the files are absent.
      say() { echo "[strata] $*"; }
      say "model directory $MODEL_PATH has no .gguf yet; waiting up to 10 min for GPUStack's download"
      i=0
      while [ $i -lt 40 ] && ! ls "$MODEL_PATH"/*.gguf >/dev/null 2>&1; do
        sleep 15
        i=$((i + 1))
      done
      if ls "$MODEL_PATH"/*.gguf >/dev/null 2>&1; then
        GGUF_DIR="$MODEL_PATH"
      else
        say "no .gguf appeared in $MODEL_PATH; Strata will download the model itself"
      fi
    fi
  elif [ -f "$MODEL_PATH" ] && [ "${MODEL_PATH##*/}" = *.gguf ]; then
    GGUF_DIR="${MODEL_PATH%/*}"
    [ -n "$GGUF_DIR" ] || GGUF_DIR="."
  fi
fi
[ -n "$GGUF_DIR" ] && echo "[strata] reading the GGUFs from $GGUF_DIR (supplied by GPUStack)"

# ---- the rest is the project's entry point ------------------------------------
export PORT
STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"
CONTEXT="${CONTEXT:-32768}"
VISION="${VISION:-no}"          # no | yes | cpu (the image encoder on the CPU)
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-}"
KV="${KV:-}"                    # int8 | q4_0 | k8v4; empty: setup.py's own default (int8)
GPUS="${GPUS:-}"                # "0,2" or "all": one model across several cards (docs/MULTI_GPU.md)
GPU="${GPU:-}"                  # one card, numbered as nvidia-smi numbers them
LAYER_SPLIT="${LAYER_SPLIT:-}"  # with GPUS: where each later card's layers start (default: auto)
LOW_RAM="${LOW_RAM:-auto}"      # on: the experts come from the pack's experts.bin, not from RAM

# setup.py starts the newest strata-*.json it finds, so link in exactly the one
# this family and model were set up with. The config is the recorded output of
# that setup (the pack, the profile, the quant, the KV decision), not settings
# the entry point could rebuild from env vars. qwen has an empty family tag.
case "$FAMILY" in qwen) prefix="" ;; *) prefix="${FAMILY}-" ;; esac
tag="${prefix}$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')"
cfg="$STRATA_DATA/config/strata-$tag.json"
mkdir -p "$STRATA_DATA/config"

# REINSTALL is only needed to change settings for a model that is already set up
# (context, vision, KV, host, api_key). Switching between models already on the
# volume needs no setup pass: their config is already there.
#
# --gguf-dir is passed only when GPUStack supplied the GGUFs; then setup.py reads
# them instead of downloading. LOW_RAM is always passed: setup.py measures the
# PC's RAM from /proc/meminfo, which in a container is the host's total, not the
# container's limit, so a memory-capped container has to ask for the low-RAM mode
# itself.
if [ "${REINSTALL:-0}" = "1" ] || [ ! -f "$cfg" ]; then
  echo "Setting up $tag (model: ${GGUF_DIR:-downloaded by Strata})."
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision "$VISION" \
    --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
    --port "$PORT" --no-start --low-ram "$LOW_RAM"
  if [ -n "$GGUF_DIR" ]; then set -- "$@" --gguf-dir "$GGUF_DIR"; fi
  if [ -n "$KV" ]; then set -- "$@" --kv "$KV"; fi
  if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
  if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
  if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
  .venv/bin/python setup.py --setup --yes "$@"
  [ -e "/opt/strata/strata-$tag.json" ] && { cmp -s "/opt/strata/strata-$tag.json" "$cfg" || cp -f "/opt/strata/strata-$tag.json" "$cfg"; }
else
  [ -e "/opt/strata/strata-$tag.json" ] || ln -s "$cfg" "/opt/strata/strata-$tag.json"
fi

# Later starts skip straight here: setup.py finds the installed config and
# launches serve/server.py (OpenAI- and Anthropic-compatible API on $PORT).
# GPUS / GPU / LAYER_SPLIT are repeated on purpose. Given at the start they pin the
# cards for this model, and setup.py saves them in its config; without them a config
# that names one card is offered once to a pair, on its own, when the host has two
# cards that can share the model (setup.py's offer_together, docs/MULTI_GPU.md).
set -- --port "$PORT"
if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
exec .venv/bin/python setup.py "$@"
