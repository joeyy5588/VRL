#!/bin/bash
set -e
python boba/1_generate_criteria.py --answer-file-folder ../code/kiki_v2_logs_72b --image-folder ../kiki_bouba_v2_split/train_evolve_mini
python boba/2_convert_criteria_caption.py --explanation-folder ../code/kiki_v2_logs_72b  --answer-file-folder ../code/kiki_v2_logs_72b
python boba/3_generate_feature_ov.py --question-file-folder ../code/kiki_v2_logs_72b --answer-file-folder ../code/kiki_v2_logs_72b --image-folder ../kiki_bouba_v2_split/train_evolve_mini
python boba/3_generate_feature_ov.py --question-file-folder ../code/kiki_v2_logs_72b --answer-file-folder ../code/kiki_v2_logs_72b_inference --image-folder ../kiki_bouba_v2_split/val_evolve_mini

python boba/a_generate_criteria_same.py --answer-file-folder ../code/kiki_v2_logs_72b_same --image-folder ../kiki_bouba_v2_split/train_evolve_mini
python boba/b_convert_criteria_caption.py --explanation-folder ../code/kiki_v2_logs_72b_same  --answer-file-folder ../code/kiki_v2_logs_72b_same
python boba/c_generate_feature_ov.py --question-file-folder ../code/kiki_v2_logs_72b_same --answer-file-folder ../code/kiki_v2_logs_72b_same --image-folder ../kiki_bouba_v2_split/train_evolve_mini
python boba/c_generate_feature_ov.py --question-file-folder ../code/kiki_v2_logs_72b_same --answer-file-folder ../code/kiki_v2_logs_72b_same_inference --image-folder ../kiki_bouba_v2_split/val_evolve_mini