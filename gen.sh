RUNID=${1}
EPOCHS=${2}
DATASET=${3:-"crosswoz"}
LANG=${4:-"en"}

for EPOCH in $EPOCHS
do
  for SPLIT in val test test-250
  do
    CKPT="ytlin/${RUNID}_${EPOCH}"
    FILENAME="ytlin_${RUNID}_${EPOCH}"
    echo "${CKPT}"
    echo "${DATASET}"
    echo "${LANG}"
    echo "${SPLIT}"
    python generate_conditional_lm_output.py "${CKPT}" "${DATASET}" "${LANG}" "${SPLIT}"
    python parse_conditional_lm_output.py "submission/crosswoz/en/val/${FILENAME}.json" "submission/crosswoz/en/val/submission.${FILENAME}.json" "${DATASET}" "${LANG}"
  done
done