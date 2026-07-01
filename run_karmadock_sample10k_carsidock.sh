#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-g12v}"
IMAGE="${CARSIDOCK_IMAGE:-carsidock:v1}"
WORK_ROOT="/home/aidd"
LIGANDS="/work/CarsiDock/inputs/karmadock_sample10k/sample10k_smiles.txt"

case "$TARGET" in
  g12d|7rpz)
    TARGET="g12d"
    PDB="/work/KarmaDock/runs/g12d_sample10k/input/g12d_protein.pdb"
    REFLIG="/work/KarmaDock/runs/g12d_sample10k/input/g12d_ligand_6IC.mol2"
    OUT="/work/CarsiDock/outputs/g12d_sample10k"
    ;;
  g12v|g12v_br7)
    TARGET="g12v"
    PDB="/work/KarmaDock/runs/g12v_sample10k/input/g12v_protein.pdb"
    REFLIG="/work/KarmaDock/runs/g12v_sample10k/input/g12v_ligand_BR7.mol2"
    OUT="/work/CarsiDock/outputs/g12v_sample10k"
    ;;
  *)
    echo "Usage: $0 {g12d|g12v}" >&2
    exit 2
    ;;
esac

mkdir -p "${WORK_ROOT}/CarsiDock/outputs/${TARGET}_sample10k"

docker run --rm \
  -v "${WORK_ROOT}:/work" \
  --gpus all \
  --shm-size 16g \
  --ulimit nofile=65535:65535 \
  -w /work/CarsiDock \
  "$IMAGE" \
  python /work/CarsiDock/run_screening.py \
    --pdb_file "$PDB" \
    --reflig "$REFLIG" \
    --ligands "$LIGANDS" \
    --output_dir "$OUT" \
    --cuda_convert
