fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "preprocess/WP_train.bpe" \
  --destdir "wp_kw_story-bin/" \
  --workers 60 \
  --srcdict "bart.large/dict.txt" \
  --tgtdict "bart.large/dict.txt";