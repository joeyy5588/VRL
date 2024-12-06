import json
import os

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
    class_1_ans = question_dict["class_1"]
    class_2_ans = question_dict["class_2"]
    if args.use_cot:
        question2 = f"Now, based on your answer to the question and the following class descriptions, determine which class the image belongs to:"
        question2 += f"\nClass 1: {class_1_ans}"
        question2 += f"\nClass 2: {class_2_ans}"

        if args.use_other_label:
            question2 += f"\nClass 3: Neither of the above"
            question2 += "\nPlease response with **only one of the following**: \"Class 1\", \"Class 2\", or \"Class 3\""
            question2 += "\nDo not include any other text in your response. Just output the class number."
        else:
            question2 += "\nPlease response with **only one of the following**: \"Class 1\" or \"Class 2\""
            question2 += "\nDo not include any other text in your response. Just output the class number."

        argument = {
            "image_file": image_fn,
            "question1": question,
            "question2": question2,
            "question_dict": question_dict,
            "question_id": question_dict["question_id"],
            "label": label,
        }

    else:
        question1 = f"Given the following image, classify it based on the provided criteria:"
        question1 += f"\nCriteria (Question): {question}"
        question1 += f"\nClass 1: {class_1_ans}"
        question1 += f"\nClass 2: {class_2_ans}"

        if args.use_other_label:
            question1 += f"\nClass 3: Neither of the above"
            question1 += "\nPlease response with \"Class 1\", \"Class 2\", or \"Class 3\""

        else:
            question1 += "\nPlease response with \"Class 1\" or \"Class 2\""

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


    question_file = f"{args.question_file_folder}/questions.jsonl"
    arguments = []
    with open(question_file, "r") as f:
        lines = f.readlines()
        question_id = 0
        for line in lines:
            data = json.loads(line)
            question_dict = data["pred"]
            question_dict["question_id"] = question_id
            class_1_name = data["class_1"]
            class_2_name = data["class_2"]
            class_1_image_list = []
            class_2_image_list = []
            class_3_image_list = []

            class1_img_folder = os.path.join(args.image_folder, class_1_name)
            for img_fn in os.listdir(class1_img_folder):
                class_1_image_list.append(os.path.abspath(os.path.join(class1_img_folder, img_fn)))
            class2_img_folder = os.path.join(args.image_folder, class_2_name)
            for img_fn in os.listdir(class2_img_folder):
                class_2_image_list.append(os.path.abspath(os.path.join(class2_img_folder, img_fn)))

            for subdir in os.listdir(args.image_folder):
                if (subdir not in [class_1_name, class_2_name]):
                    curr_img_folder = os.path.join(args.image_folder, subdir)
                    for img_fn in os.listdir(curr_img_folder):
                        class_3_image_list.append(os.path.abspath(os.path.join(curr_img_folder, img_fn)))

            for img_fn in class_1_image_list:
                argument = create_question_prompt(args, question_dict, img_fn, 1)
                arguments.append(argument)

            for img_fn in class_2_image_list:
                argument = create_question_prompt(args, question_dict, img_fn, 2)
                arguments.append(argument)

            for img_fn in class_3_image_list:
                argument = create_question_prompt(args, question_dict, img_fn, 3)
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
    id2acc = {}
    id2question = {}
    for s, state in enumerate(states):
        if args.use_cot:
            pred_text = state["answer2"].strip().lower()
        else:
            pred_text = state["answer"].strip().lower()

        if args.use_other_label:

            if pred_text.startswith("class 1") or (pred_text.count("class 1") > pred_text.count("class 2") and pred_text.count("class 1") > pred_text.count("class 3"))\
                or (pred_text.count("1") > pred_text.count("2") and pred_text.count("1") > pred_text.count("3")):
                pred = 1
            elif pred_text.startswith("class 2") or (pred_text.count("class 2") > pred_text.count("class 1") and pred_text.count("class 2") > pred_text.count("class 3"))\
                or (pred_text.count("2") > pred_text.count("1") and pred_text.count("2") > pred_text.count("3")):
                pred = 2
            elif pred_text.startswith("class 3") or (pred_text.count("class 3") > pred_text.count("class 1") and pred_text.count("class 3") > pred_text.count("class 2"))\
                or (pred_text.count("3") > pred_text.count("1") and pred_text.count("3") > pred_text.count("2")):
                pred = 3
            else:
                print('Invalid prediction', pred_text)
                # pred = random.choice([1, 2, 3])
                pred = 3

        else:
            if pred_text.startswith("class 1") or pred_text.count("class 1") > pred_text.count("class 2")\
                or pred_text.count("1") > pred_text.count("2"):
                pred = 1
            elif pred_text.startswith("class 2") or pred_text.count("class 2") > pred_text.count("class 1")\
                or pred_text.count("2") > pred_text.count("1"):
                pred = 2
            else:
                print('Invalid prediction', pred_text)
                pred = random.choice([1, 2])

        preds.append(pred)


    answer_file_fn = f"{args.answer_file_folder}/features_{args.use_cot}_{args.use_other_label}.jsonl"
    print(f"Writing to {answer_file_fn}")
    with open(answer_file_fn, "w") as f:
        for i, pred in enumerate(preds):
            output_dict = {
                "question_dict": arguments[i]["question_dict"],
                "question_id": arguments[i]["question_id"],
                "pred": pred,
                "image_file": arguments[i]["image_file"]
            }
            if args.use_cot:
                output_dict["answer"] = states[i]["answer"]
            f.write(json.dumps(output_dict) + "\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file-folder", type=str, default="sglang_questions")
    parser.add_argument("--answer-file-folder", type=str, default="sglang_inference_features")
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

