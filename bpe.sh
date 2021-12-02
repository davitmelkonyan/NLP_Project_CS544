for SPLIT in train valid
do
  for LANG in source target
  do
    python -m multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "base_project/Data_/WP/WP_LM/WP_$SPLIT.$LANG" \
    --outputs "preprocess/WP_$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

#for SPLIT in train valid
#do
#  for LANG in source target
#  do
#    python -m multiprocessing_bpe_encoder \
#    --encoder-json encoder.json \
#    --vocab-bpe vocab.bpe \
#    --inputs "base_project/Data_/ROC/ROC_LM/ROC_$SPLIT.$LANG" \
#    --outputs "preprocess/ROC_$SPLIT.bpe.$LANG" \
#    --workers 60 \
#    --keep-empty;
#  done
#done