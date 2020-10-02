RUNID=${1}
EPOCHS=${2}
DATASET=${3:-"crosswoz"}
LANG=${4:-"en"}

for EPOCH in $EPOCHS
do
  for SPLIT in val human_val test data
  do
    CKPT="${RUNID}_${EPOCH}"
    DATA="./data/${DATASET}/processed/${LANG}/${SPLIT}.json"
    echo "${CKPT}"
    echo "${DATA}"
    python make_submission.py "ytlin/${CKPT}" "${CKPT}" --test_set "${DATA}"
#    python parse_output.py "${DATA}.${CKPT}"
  done
done