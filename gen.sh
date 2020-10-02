RUNID=${1}
DATASET=${2:-"crosswoz"}
LANG=${3:-"en"}

for EPOCH in 1 6 9
do
  for SPLIT in val human_val test data
  do
    CKPT="${RUNID}_${EPOCH}"
    DATA="./data/${DATASET}/processed/${LANG}/${SPLIT}.json"
    echo "${CKPT}"
    echo "${DATA}"
    python make_submission.py "ytlin/${CKPT}" "${CKPT}" --test_set "${DATA}"
    python parse_output.py "${DATA}.${CKPT}"
  done
done