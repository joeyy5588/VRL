import json
import os

from inat_utils import category_list, super_class_list
import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
import argparse
import concurrent.futures

import base64
import openai
from tqdm import tqdm
import random

def create_openai_client(base_url):
    return openai.Client(api_key="EMPTY", base_url=base_url)

def encode_image(imag_path):
    with open(imag_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def multi_image_stream_request_test(client, image, question1, question2=None, max_tokens=256, use_cot=False):
    image = encode_image(image)

    response = client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": question1,
                    },
                ],
            },
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    answer = response.choices[0].message.content
    answer2 = None
    if use_cot:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": question1,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question2,
                        },
                    ],
                },
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        answer2 = response.choices[0].message.content

    return {"answer": answer, "answer2": answer2}

def process_single_image_stream(i, arguments, client, use_cot=False):
    state = multi_image_stream_request_test(client, arguments[i]["image_file"], arguments[i]["question1"], arguments[i]["question2"], use_cot=use_cot)
    return i, state

def create_question_prompt(args, question_dict, image_fn, label):
    question = question_dict["question"]
    question1 = f"Examine the given image and determine if it matches the features described by the following criteria: {question}"
    question1 += "Answer only with YES or NO." 

    argument = {
        "image_file": image_fn,
        "question1": question1,
        "question2": None,
        "question_dict": question_dict,
        "question_id": question_dict["question_id"],
        "label": label,
    }

    return argument


@sgl.function
def image_qa(s, image_file, question, **kwargs):
    s += sgl.user(sgl.image(image_file) + question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=64))

def main(args):
    if not os.path.exists(args.answer_file_folder):
        os.makedirs(args.answer_file_folder)

    for cat, super_class in zip(category_list, super_class_list):
        print(f"Processing {cat}")

        question_file = f"{args.question_file_folder}/{cat}_questions.jsonl"
        arguments = []
        with open(question_file, "r") as f:
            lines = f.readlines()
            print(f"Processing {len(lines)} questions")
            question_id = 0
            for line in lines:
                data = json.loads(line)
                assert question_id == data["question_id"]
                question_dict = {}
                question_dict["question"] = data["pred"]
                question_dict["question_id"] = data["question_id"]
                class_name = data["class_name"]
                class_1_image_list = []
                class_2_image_list = []

                for subdir in os.listdir(args.image_folder):
                    if class_name in subdir:
                        class1_img_folder = os.path.join(args.image_folder, subdir)
                        for img_fn in os.listdir(class1_img_folder):
                            class_1_image_list.append(os.path.abspath(os.path.join(class1_img_folder, img_fn)))
                for super_class_name in super_class:
                    for subdir in os.listdir(args.image_folder):
                        if super_class_name in subdir and (super_class_name != class_name):
                            curr_img_folder = os.path.join(args.image_folder, subdir)
                            for img_fn in os.listdir(curr_img_folder):
                                class_2_image_list.append(os.path.abspath(os.path.join(curr_img_folder, img_fn)))

                for img_fn in class_1_image_list:
                    argument = create_question_prompt(args, question_dict, img_fn, 1)
                    arguments.append(argument)

                for img_fn in class_2_image_list:
                    argument = create_question_prompt(args, question_dict, img_fn, 2)
                    arguments.append(argument)

                question_id += 1

        states = [None] * len(arguments)

        # Select backend
        client = create_openai_client("http://127.0.0.1:30000/v1")

        # single-threaded
        if args.threads == 1:
            for i in tqdm(range(len(arguments))):
                state = multi_image_stream_request_test(client, arguments[i]["image_file"], arguments[i]["question1"], arguments[i]["question2"], use_cot=args.use_cot)
                states[i] = state
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
                futures = [executor.submit(process_single_image_stream, i, arguments, client, use_cot=args.use_cot) for i in range(len(arguments))]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Images"):
                    i, state = future.result()
                    states[i] = state
            

        preds = []
        response_text = []
        id2acc = {}
        id2question = {}
        for s, state in enumerate(states):
            pred_text = state["answer"].strip().lower()
            response_text.append(pred_text)
            if pred_text.startswith("yes") or pred_text.count("yes") > pred_text.count("no"):
                pred = 1
            elif pred_text.startswith("no") or pred_text.count("no") > pred_text.count("yes"):
                pred = 2
            else:
                print('Invalid prediction', pred_text)
                pred = random.choice([1, 2])

            preds.append(pred)
            gt = arguments[s]["label"]
            question_id = arguments[s]["question_dict"]["question_id"]
            if question_id not in id2acc:
                id2acc[question_id] = {"total": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
                id2question[question_id] = arguments[s]["question_dict"]

            id2acc[question_id]["total"] += 1
            if pred == gt:
                id2acc[question_id]["correct"] += 1
            if pred == 1 and gt == 1:
                id2acc[question_id]["tp"] += 1
            elif pred == 1 and gt == 2:
                id2acc[question_id]["fp"] += 1
            elif pred == 2 and gt == 1:
                id2acc[question_id]["fn"] += 1

    
        answer_file_fn = f"{args.answer_file_folder}/{cat}_questions_{args.use_other_label}.jsonl"
        print(f"Writing to {answer_file_fn}")
        with open(answer_file_fn, "w") as f:
            for id, question_dict in id2question.items():
                precision = id2acc[id]["tp"]/(id2acc[id]["tp"] + id2acc[id]["fp"]) if (id2acc[id]["tp"] + id2acc[id]["fp"]) > 0 else 0
                recall = id2acc[id]["tp"]/(id2acc[id]["tp"] + id2acc[id]["fn"]) if (id2acc[id]["tp"] + id2acc[id]["fn"]) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                output_dict = {
                    "question_dict": question_dict,
                    "accuracy": id2acc[question_dict["question_id"]]["correct"]/id2acc[question_dict["question_id"]]["total"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "question_id": question_dict["question_id"],
                }
                f.write(json.dumps(output_dict) + "\n")
    
        answer_file_fn = f"{args.answer_file_folder}/{cat}_features_{args.use_cot}_{args.use_other_label}.jsonl"
        print(f"Writing to {answer_file_fn}")
        with open(answer_file_fn, "w") as f:
            for i, pred in enumerate(preds):
                output_dict = {
                    "question_dict": arguments[i]["question_dict"],
                    "question_id": arguments[i]["question_id"],
                    "pred": pred,
                    "response_text": response_text[i],
                    "image_file": arguments[i]["image_file"]
                }
                f.write(json.dumps(output_dict) + "\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file-folder", type=str, default="sglang_questions_72b_same")
    parser.add_argument("--answer-file-folder", type=str, default="sglang_features_72b_same")
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

