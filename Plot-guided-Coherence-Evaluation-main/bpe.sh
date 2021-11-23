for LANG in source target
do
  python -m multiprocessing_bpe_encoder \
  --encoder-json encoder.json \
  --vocab-bpe vocab.bpe \
  --inputs "Data_/WP/WP_LM/WP_train.$LANG" \
  --outputs "preprocess/WP_train.bpe.$LANG" \
  --workers 60 \
  --keep-empty;
done