from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from inat_utils import category_list, super_class_list
import argparse
import numpy as np
import json
import os
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(),
    "Naive Bayes": BernoulliNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP Classifier (Neural Network)": MLPClassifier()
}

def process_feature(feature_file, super_class, args):
    id2question = {}
    id2question_dict = {}
    img2feature = {}
    img2label = {}
    with open(feature_file, "r") as f:
        lines = f.readlines()

        last_line = json.loads(lines[-1])

        num_question = last_line["question_id"] + 1
        if args.use_other_label:
            num_question *= 2

        for line in lines:
            data = json.loads(line)

            question_id = data["question_id"]
            img_fn = data["image_file"]
            img_class_name = img_fn.split("/")[-2]
            img_class_name = img_class_name.split("_")[-2] + "_" + img_class_name.split("_")[-1]
            label = super_class.index(img_class_name)
            img2label[img_fn] = label
            pred = data["pred"]
            question_dict = data["question_dict"]

            id2question[question_id] = question_dict["question"]# + f"\nClass 1: {question_dict['class_1']}\nClass 2: {question_dict['class_2']}"
            id2question_dict[question_id] = question_dict
            
            if img_fn not in img2feature:
                img2feature[img_fn] = np.zeros((num_question))
            
            if pred == 1:
                img2feature[img_fn][question_id] = 0
            elif pred == 2:
                img2feature[img_fn][question_id] = 1
            elif pred == 3:
                img2feature[img_fn][question_id + num_question//2] = 1

    # fill the filtered question id with blank sentences
    for i in range(num_question):
        if i not in id2question:
            id2question[i] = ""

    # sort id2question
    id2question = dict(sorted(id2question.items(), key=lambda x: x[0]))

    train_x = list(img2feature.values())
    train_y = list(img2label.values())
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return train_x, train_y, id2question


def main(args):
    model2result = {}
    for model, clf in models.items():
        model2result[model] = {}

        for cat, super_class in zip(category_list, super_class_list):
            feature_file = args.training_feature_file
            inference_feature_file = args.inference_feature_file
            train_x, train_y, id2question = process_feature(feature_file, super_class, args)
            inference_x, inference_y, _ = process_feature(inference_feature_file, super_class, args)
            train_feature_dim = train_x.shape[1]
            inference_x = inference_x[:, :train_feature_dim]
            feature_folder = feature_file.split("/")[0]
            inference_feature_folder = inference_feature_file.split("/")[0]

            np.save(f"{feature_folder}/{cat}_features_llava.npy", train_x)
            np.save(f"{feature_folder}/{cat}_labels_llava.npy", train_y)
            np.save(f"{inference_feature_folder}/{cat}_features_llava.npy", inference_x)
            np.save(f"{inference_feature_folder}/{cat}_labels_llava.npy", inference_y)

            clf.fit(train_x, train_y)
            pred_train_y = clf.predict(train_x)
            pred_inference_y = clf.predict(inference_x)

            model2result[model][cat] = {
                "train_accuracy": accuracy_score(train_y, pred_train_y),
                "inference_accuracy": accuracy_score(inference_y, pred_inference_y),            
            }
        
        model2result[model]["average"] = {
            "train_accuracy": np.mean([result["train_accuracy"] for result in model2result[model].values()]),
            "inference_accuracy": np.mean([result["inference_accuracy"] for result in model2result[model].values()]),
        }

        print(f"Model: {model}")
        print(f"Average Train Accuracy: {model2result[model]['average']['train_accuracy']}")
        print(f"Average Inference Accuracy: {model2result[model]['average']['inference_accuracy']}")

    os.makedirs(args.result_folder, exist_ok=True)
    with open(f"{args.result_folder}/{args.result_file}", "w") as f:
        json.dump(model2result, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-feature-file", type=str, default="sglang_features/fungi_features.jsonl")
    parser.add_argument("--inference-feature-file", type=str, default="sglang_inference_features/fungi_features.jsonl")
    parser.add_argument("--result-file", type=str, default="results.json")
    parser.add_argument("--result-folder", type=str, default="sklearn_results")
    parser.add_argument("--use_other_label", action="store_true", default=False)
    parser.add_argument("--use_cot", action="store_true", default=False)
    args = parser.parse_args()

    main(args)