#!/bin/bash
set -e
# python 3_generate_feature_ov.py 
# python 3_generate_feature_ov.py --answer-file-folder sglang_features --image-folder train_evolve_mini
# python 3_generate_feature_ov.py --answer-file-folder sglang_features_72b --image-folder train_evolve_mini --use-cot --use-other-label

# python c_generate_feature_ov.py 
# python c_generate_feature_ov.py --answer-file-folder sglang_features_72b_same_inference --image-folder val

# 7b
# CUDA_VISIBLE_DEVICES=4 python 3_generate_feature_clip.py --answer-file-folder sglang_features --image-folder train_evolve_mini --question-file-folder sglang_questions
# CUDA_VISIBLE_DEVICES=4 python 3_generate_feature_clip.py --answer-file-folder sglang_inference_features --image-folder val --question-file-folder sglang_questions
CUDA_VISIBLE_DEVICES=4 python c_generate_feature_clip.py --answer-file-folder sglang_features_7b_same --image-folder train_evolve_mini --question-file-folder sglang_questions_7b_same
CUDA_VISIBLE_DEVICES=4 python c_generate_feature_clip.py --answer-file-folder sglang_features_7b_same_inference --image-folder val --question-file-folder sglang_questions_7b_same

# 72b
# CUDA_VISIBLE_DEVICES=4 python 3_generate_feature_clip.py --answer-file-folder sglang_features_72b --image-folder train_evolve_mini --question-file-folder sglang_questions_72b
# CUDA_VISIBLE_DEVICES=4 python 3_generate_feature_clip.py --answer-file-folder sglang_inference_features_72b --image-folder val --question-file-folder sglang_questions_72b
CUDA_VISIBLE_DEVICES=4 python c_generate_feature_clip.py --answer-file-folder sglang_features_72b_same --image-folder train_evolve_mini --question-file-folder sglang_questions_72b_same
CUDA_VISIBLE_DEVICES=4 python c_generate_feature_clip.py --answer-file-folder sglang_features_72b_same_inference --image-folder val --question-file-folder sglang_questions_72b_same
