fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "preprocess/WP_train.bpe" \
  --validpref "preprocess/WP_valid.bpe" \
  --destdir "wp_kw_story-bin/" \
  --workers 60 \
  --srcdict "bart.large/dict.txt" \
  --tgtdict "bart.large/dict.txt";

#fairseq-preprocess \
#  --source-lang "source" \
#  --target-lang "target" \
#  --trainpref "preprocess/ROC_train.bpe" \
#  --validpref "preprocess/ROC_valid.bpe" \
#  --destdir "roc_kw_story-bin/" \
#  --workers 60 \
#  --srcdict "bart.large/dict.txt" \
#  --tgtdict "bart.large/dict.txt";