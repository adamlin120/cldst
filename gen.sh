RUNID=${1}
EPOCHS=${2}
DATASET=${3:-"crosswoz"}
LANG=${4:-"en"}

for EPOCH in $EPOCHS
do
  for SPLIT in val test data
  do
    CKPT="${RUNID}_${EPOCH}"
    DATA="./data/${DATASET}/processed/${LANG}/${SPLIT}.json"
    echo "${CKPT}"
    echo "${DATA}"
    python generate_conditional_lm_output.py "ytlin/${CKPT}" "${CKPT}" "${DATA}"
  done
done