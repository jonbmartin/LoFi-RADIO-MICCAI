# Shared helpers for the experiment scripts. Source from the repository root:
#   source scripts/common.sh
#
# Environment (all optional):
#   DINO_W    local DINOv2 ViT-S/14 weights  (default dinov2_vits14.safetensors; hub download if absent)
#   PHIKON_W  local Phikon weights           (default phikon/model.safetensors;  hub download if absent)
#   CUDA_VISIBLE_DEVICES   GPUs used for distillation (DataParallel over all visible devices)
#   DISTILL_BATCH / DISTILL_EPOCHS   distillation batch (default 256, for 4 GPUs) and epochs (default 40)
set -e
DINO_W=${DINO_W:-dinov2_vits14.safetensors}
PHIKON_W=${PHIKON_W:-phikon/model.safetensors}
DISTILL_BATCH=${DISTILL_BATCH:-256}
DISTILL_EPOCHS=${DISTILL_EPOCHS:-40}
SEEDS=${SEEDS:-0,1,2,3,4,5,6,7,8,9}
mkdir -p cache runs

# "name=path" list of the teacher weights that exist locally (others load from the hub)
teacher_weights () {
  local w=""
  [ -f "$DINO_W" ]   && w="dinov2=$DINO_W"
  [ -e "$PHIKON_W" ] && w="${w:+$w,}phikon=$PHIKON_W"
  echo "$w"
}

# dense_cache TAG SOURCES [TEACHERS]  ->  cache/dense_TAG.npz   (skipped if present; tens of GB)
dense_cache () {
  local tag="$1" sources="$2" teachers="${3:-dinov2,biomedclip,phikon}"
  local out="cache/dense_${tag}.npz"
  if [ -f "$out" ]; then echo "  [dense $tag] exists -- skip."; return; fi
  echo "==== dense teacher cache '$tag': $teachers over $sources ===="
  python lofi_radio/cache_dense.py --sources "$sources" --teachers "$teachers" \
    --weights "$(teacher_weights)" --out "$out"
}

# student NAME DENSE_TAG [extra distill.py args...]
#   ->  runs/student_NAME.pt  +  cache/student_NAME_cls_lisa.npz   (skipped if the export exists)
student () {
  local name="$1" tag="$2"; shift 2
  local ckpt="runs/student_${name}.pt" feat="cache/student_${name}_cls_lisa.npz"
  if [ -f "$feat" ]; then echo "  [student $name] exists -- skip."; return; fi
  echo "==== distill student '$name' from cache/dense_${tag}.npz ===="
  python lofi_radio/distill.py --cache "cache/dense_${tag}.npz" \
    ${DINO_W:+$([ -f "$DINO_W" ] && echo --weights "$DINO_W")} \
    --batch "$DISTILL_BATCH" --epochs "$DISTILL_EPOCHS" --out "$ckpt" "$@"
  python lofi_radio/export_student.py --ckpt "$ckpt" --out "$feat"
}

# frozen_cache NAME [extra cache_teachers.py args...]  ->  cache/NAME_lisa.npz   (skipped if present)
frozen_cache () {
  local name="$1"; shift
  local out="cache/${name}_lisa.npz"
  if [ -f "$out" ]; then echo "  [$name] cache exists -- skip."; return 0; fi
  echo "---- cache $name ----"
  python lofi_radio/cache_teachers.py --teacher "$name" --sources lisa --out "$out" "$@"
}
