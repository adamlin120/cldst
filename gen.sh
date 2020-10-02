RUNID="1cejkgok"
for EPOCH in 1 6 9
do
  for SPLIT in test val human_val test data
  do
    CKPT="${RUNID}_${EPOCH}"
    DATA="./data/multiwoz/processed/zh/${SPLIT}.json"
    echo "${CKPT}"
    echo "${DATA}"
    python make_submission.py "ytlin/${CKPT}" "${CKPT}" --test_set "${DATA}"
    python parse_output.py "${DATA}.${CKPT}"
  done
done