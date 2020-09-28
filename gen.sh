RUNID="1cejkgok"
for EPOCH in 1 6 9
do
  for SPLIT in test val human_val test data
  do
    echo  "ytlin/${RUNID}_${EPOCH}"
    echo  "${RUNID}_${EPOCH}"
    echo  "./data/multiwoz/processed/zh/${SPLIT}.json"
    python make_submission.py "ytlin/${RUNID}_${EPOCH}" "${RUNID}_${EPOCH}" --test_set "./data/multiwoz/processed/zh/${SPLIT}.json"
  done
done