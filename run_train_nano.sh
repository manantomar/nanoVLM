#!/usr/bin/env bash
set -euo pipefail

# --- Repo root (mounted at /mnt/home) ---
REPO_ROOT="/mnt/home"
echo "REPO_ROOT=$REPO_ROOT"
cd "$REPO_ROOT"

# --- Python env ---
# if [[ -f "venv/bin/activate" ]]; then
#   source venv/bin/activate
# else
#   python3 -m pip install --upgrade pip
#   pip install -e .[train]
# fi

# --- Cluster / Volcano discovery ---
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

VC_HOSTS_FILE="$(ls /etc/volcano/*HOSTS 2>/dev/null | head -n1 || true)"
VC_NUM_FILE="$(ls /etc/volcano/*NUM 2>/dev/null | head -n1 || true)"

if [[ -n "${VC_HOSTS_FILE:-}" ]]; then
  IFS=$'\n' read -r -d '' -a HOSTS < <(tr ', ' '\n' < "$VC_HOSTS_FILE" | sed '/^$/d' && printf '\0')
  MASTER_ADDR="${HOSTS[0]}"
  NUM_NODES="${#HOSTS[@]}"
else
  MASTER_ADDR="${MASTER_ADDR:-$(hostname -f)}"
  if [[ -n "${VC_NUM_FILE:-}" ]]; then
    NUM_NODES="$(cat "$VC_NUM_FILE")"
  else
    NUM_NODES="${NUM_NODES:-1}"
  fi
fi

NODE_RANK="${VC_TASK_INDEX:-${OMPI_COMM_WORLD_RANK:-}}"
if [[ -z "$NODE_RANK" ]]; then
  echo "ERROR: Could not determine NODE_RANK"; exit 1
fi

echo "NUM_NODES=$NUM_NODES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "NODE_RANK=$NODE_RANK"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"

# --- Data path (persistent) ---
AMLT_DATA_DIR="/mnt/data/"
if [[ ! -d "$AMLT_DATA_DIR" ]]; then
  echo "ERROR: Expected data directory not found at $AMLT_DATA_DIR"; exit 1
fi

# --- NCCL / networking ---
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1

# --- Launch ---
# torchrun \
#   --nproc_per_node="${GPUS_PER_NODE}" \
#   --nnodes="${NUM_NODES}" \
#   --node_rank="${NODE_RANK}" \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
#   train.py --config "$CFG"  --data_dir "$DION_DATA_DIR"


torchrun \
    --standalone \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    train.py config/train_gpt2.py