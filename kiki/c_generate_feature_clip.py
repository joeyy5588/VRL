import json
import os

from kiki_utils import category_list_v1, category_list_v2
import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
import argparse
from tqdm import tqdm
import random
import open_clip
import torch
from PIL import Image
import numpy as np
def main(args):
    if not os.path.exists(args.answer_file_folder):
        os.makedirs(args.answer_file_folder)

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    question_file = f"{args.question_file_folder}/questions.jsonl"
    arguments = []
    image_fn_list = []
    label_list = []
    text_attr_list = []
    with open(question_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)

            class_attr = data["pred"]

            text_attr_list.append(class_attr)

    category_list = category_list_v1 if "v1" in args.question_file_folder else category_list_v2
    for subdir in os.listdir(args.image_folder):
        class_name = subdir
        curr_img_folder = os.path.join(args.image_folder, subdir)
        for img_fn in os.listdir(curr_img_folder):
            image_fn_list.append(os.path.abspath(os.path.join(curr_img_folder, img_fn)))
            label_list.append(category_list.index(class_name))

    image_list = [preprocess(Image.open(img_fn).convert("RGB")) for img_fn in image_fn_list]
    images = torch.tensor(np.stack(image_list)).to(device)

    text_tokens = tokenizer(text_attr_list).to(device)

    print(images.shape)
    print(text_tokens.shape)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # get the similarity of the image and text features
        similarity = (image_features @ text_features.T).squeeze()  # Cosine similarity between image and text 1
        print(similarity.shape)

        # get the binary feature
        binary_feature = (similarity < 0.2).int()
        binary_feature = binary_feature.cpu().numpy()
        print(binary_feature.shape)

        continous_feature = similarity.cpu().numpy()
        print(continous_feature.shape)

    y_label = np.array(label_list)
    print(y_label.shape)

    # save the feature and label
    np.save(f"{args.answer_file_folder}/continous_features.npy", continous_feature)
    np.save(f"{args.answer_file_folder}/binary_features.npy", binary_feature)
    np.save(f"{args.answer_file_folder}/labels.npy", y_label)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file-folder", type=str, default="sglang_questions")
    parser.add_argument("--answer-file-folder", type=str, default="sglang_features")
    parser.add_argument("--image-folder", type=str, default="train_evolve_mini")
    parser.add_argument("--use-cot", action="store_true", default=False)
    parser.add_argument("--use-other-label", action="store_true", default=False)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

