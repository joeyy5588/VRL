python boba/cache_train_test.py --result-folder ../code/kiki_v1_results --result-file 72b_contrastive --train-feature-file ../code/kiki_v1_logs_72b/features_False_False.jsonl --inference-feature-file ../code/kiki_v1_logs_72b_inference/features_False_False.jsonl
python boba/cache_train_test.py --result-folder ../code/kiki_v1_results --result-file 72b_same --train-feature-file ../code/kiki_v1_logs_72b_same/features_False_False.jsonl --inference-feature-file ../code/kiki_v1_logs_72b_same_inference/features_False_False.jsonl
python boba/cache_train_test_merge.py --result-folder ../code/kiki_v1_results --result-file 72b_concat \
--train-feature-file ../code/kiki_v1_logs_72b/features_False_False.jsonl --inference-feature-file ../code/kiki_v1_logs_72b_inference/features_False_False.jsonl \
--train-feature-file-2 ../code/kiki_v1_logs_72b_same/features_False_False.jsonl --inference-feature-file-2 ../code/kiki_v1_logs_72b_same_inference/features_False_False.jsonl