import json
import os

from inat_utils import category_list, super_class_list
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(),
    "Naive Bayes": BernoulliNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP Classifier (Neural Network)": MLPClassifier(),
    "Voting Classifier": VotingClassifier(estimators=[
        ("lr", LogisticRegression()), ("rf", RandomForestClassifier()), ("mlp", MLPClassifier())
    ], voting="soft"),
    "Stacking Classifier": StackingClassifier(estimators=[
        ("lr", LogisticRegression()), ("rf", RandomForestClassifier()), ("mlp", MLPClassifier())
    ], final_estimator=LogisticRegression())
}

def main(args):
    if not os.path.exists(args.answer_file_folder):
        os.makedirs(args.answer_file_folder)

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    cat2features = {}
    cat2labels = {}
    for cat, super_class in zip(category_list, super_class_list):
        print(f"Processing {cat}")

        question_file = f"{args.question_file_folder}/{cat}_questions.jsonl"
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

        for subdir in os.listdir(args.image_folder):
            class_name = subdir.split("_")[-2] + "_" + subdir.split("_")[-1]
            if class_name in super_class:
                curr_img_folder = os.path.join(args.image_folder, subdir)
                for img_fn in os.listdir(curr_img_folder):
                    image_fn_list.append(os.path.abspath(os.path.join(curr_img_folder, img_fn)))
                    label_list.append(super_class.index(class_name))

        image_list = [preprocess(Image.open(img_fn).convert("RGB")) for img_fn in image_fn_list]
        images = torch.tensor(np.stack(image_list)).to(device)


        with torch.no_grad():
            image_features = model.encode_image(images)

            image_features /= image_features.norm(dim=-1, keepdim=True)

            image_features = image_features.cpu().numpy()

        
        y_label = np.array(label_list)

        cat2features[cat] = image_features
        cat2labels[cat] = y_label

    return cat2features, cat2labels
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file-folder", type=str, default="sglang_questions_7b_same")
    parser.add_argument("--answer-file-folder", type=str, default="sglang_features")
    parser.add_argument("--train-image-folder", type=str, default="train")
    parser.add_argument("--val-image-folder", type=str, default="val")
    parser.add_argument("--image-folder", type=str, default="train")
    parser.add_argument("--train-feature-folder-1", type=str, default="sglang_features")
    parser.add_argument("--val-feature-folder-1", type=str, default="sglang_inference_features")
    parser.add_argument("--train-feature-folder-2", type=str, default="sglang_features_7b_same")
    parser.add_argument("--val-feature-folder-2", type=str, default="sglang_features_7b_same_inference")
    parser.add_argument("--use-cot", action="store_true", default=False)
    parser.add_argument("--use-other-label", action="store_true", default=False)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = add_common_sglang_args_and_parse(parser)

    args.image_folder = args.train_image_folder
    train_x_cat, train_y_cat = main(args)
    args.image_folder = args.val_image_folder
    inference_x_cat, inference_y_cat = main(args)

    feature_postfix = "_features_llava"
    soft_voting = True
    for cat in train_x_cat.keys():
        train_x = train_x_cat[cat]
        train_y = train_y_cat[cat]
        inference_x = inference_x_cat[cat]
        inference_y = inference_y_cat[cat]
        train_x_2 = np.load(f"{args.train_feature_folder_1}/{cat}{feature_postfix}.npy")
        inference_x_2 = np.load(f"{args.val_feature_folder_1}/{cat}{feature_postfix}.npy")
        train_x_3 = np.load(f"{args.train_feature_folder_2}/{cat}{feature_postfix}.npy")
        inference_x_3 = np.load(f"{args.val_feature_folder_2}/{cat}{feature_postfix}.npy")

        # clf_1 = MLPClassifier()
        # clf_2 = VotingClassifier(estimators=[
        #     ("lr", LogisticRegression()), ("rf", RandomForestClassifier()), ("mlp", MLPClassifier())
        # ], voting="soft")
        # clf_3 = VotingClassifier(estimators=[
        #     ("lr", LogisticRegression()), ("rf", RandomForestClassifier()), ("mlp", MLPClassifier())
        # ], voting="soft")
        clf_1 = MLPClassifier()
        clf_2 = LogisticRegression()
        clf_3 = LogisticRegression()

        clf_1.fit(train_x, train_y)
        clf_2.fit(train_x_2, train_y)
        clf_3.fit(train_x_3, train_y)

        if soft_voting:
            pred_y_1 = clf_1.predict_proba(inference_x)
            pred_y_2 = clf_2.predict_proba(inference_x_2)
            pred_y_3 = clf_3.predict_proba(inference_x_3)

            pred_y = (pred_y_1 + pred_y_2 + pred_y_3) / 3
            pred_inference_y = np.argmax(pred_y, axis=1)
        else:
            pred_y_1 = clf_1.predict(inference_x)
            pred_y_2 = clf_2.predict(inference_x_2)
            pred_y_3 = clf_3.predict(inference_x_3)

            # majority voting
            pred_inference_y = []
            for y1, y2, y3 in zip(pred_y_1, pred_y_2, pred_y_3):
                if y1 == y2:
                    pred_inference_y.append(y1)
                elif y1 == y3:
                    pred_inference_y.append(y1)
                elif y2 == y3:
                    pred_inference_y.append(y2)
                else:
                    pred_inference_y.append(y1)
            
        
        print(f"Category: {cat}")
        # print(f"Train Accuracy: {accuracy_score(train_y, pred_train_y)}")
        print(f"Inference Accuracy: {accuracy_score(inference_y, pred_inference_y)}")
