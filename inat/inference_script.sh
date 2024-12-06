#!/bin/bash
set -e
# 7b
python evaluate.py --result-file 7b_contrastive.json --train-feature-folder sglang_features --inference-feature-folder sglang_inference_features
python evaluate.py --result-file 7b_same.json --train-feature-folder sglang_features_7b_same --inference-feature-folder sglang_features_7b_same_inference
python evaluate.py --result-file 7b_concat.json --train-feature-folder sglang_features --inference-feature-folder sglang_inference_features --train-feature-folder-2 sglang_features_7b_same --inference-feature-folder-2 sglang_features_7b_same_inference

# 72b
python evaluate.py --result-file 72b_contrastive.json --train-feature-folder sglang_features_72b --inference-feature-folder sglang_inference_features_72b
python evaluate.py --result-file 72b_same.json --train-feature-folder sglang_features_72b_same --inference-feature-folder sglang_features_72b_same_inference
python evaluate.py --result-file 72b_concat.json --train-feature-folder sglang_features_72b --inference-feature-folder sglang_inference_features_72b --train-feature-folder-2 sglang_features_72b_same --inference-feature-folder-2 sglang_features_72b_same_inference