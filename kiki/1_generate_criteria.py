import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text, read_jsonl
import random
import argparse
from itertools import combinations
import json
import os
import pickle
from tqdm import tqdm
from kiki_utils import combinations_list_v1, combinations_list_v2
import base64
import openai
def create_openai_client(base_url):
    return openai.Client(api_key="EMPTY", base_url=base_url)

def encode_image(imag_path):
    with open(imag_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def multi_image_stream_request_test(client, image1, image2, question):
    image1 = encode_image(image1)
    image2 = encode_image(image2)

    response = client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image1}"
                        },
                        "modalities": "multi-images",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image2}"
                        },
                        "modalities": "multi-images",
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ],
        temperature=0.7,
        max_tokens=256,
    )

    return response.choices[0].message.content

@sgl.function
def image_difference_qa(s, image_1, image_2, question, **kwargs):
    s += sgl.user(sgl.image(image_1) + sgl.image(image_2) + question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=20))



def main(args):
    if not os.path.exists(args.answer_file_folder):
        os.makedirs(args.answer_file_folder)

    if "v1" in args.image_folder:
        combinations_list = combinations_list_v1
    elif "v2" in args.image_folder:
        combinations_list = combinations_list_v2

    for class_combiniation in combinations_list:
        arguments = []
        metadata_list = []
        for c in class_combiniation:
            class_1 = c[0]
            class_2 = c[1]
            class1_img_folder = os.path.join(args.image_folder, c[0])
            class2_img_folder = os.path.join(args.image_folder, c[1])

            for num in range(args.sample_per_pair):
                class1_img_fn = random.choice(os.listdir(class1_img_folder))
                class2_img_fn = random.choice(os.listdir(class2_img_folder))
                arguments.append({
                    "image_1": os.path.abspath(os.path.join(class1_img_folder, class1_img_fn)),
                    "image_2": os.path.abspath(os.path.join(class2_img_folder, class2_img_fn)),
                    "question": f"Identify the most distinctive feature that can be used to distinguish the object between image 1 and image 2.",
                    "class_1": class_1,
                    "class_2": class_2
                })

                
        states = [None] * len(arguments)

        client = create_openai_client(f"http://127.0.0.1:30000/v1")

        for i in tqdm(range(len(arguments))):
            generation_success = False
            while generation_success == False:
                state = multi_image_stream_request_test(client, arguments[i]["image_1"], arguments[i]["image_2"], arguments[i]["question"])
                if 'image 1' in state and 'image 2' in state:
                    generation_success = True
            states[i] = state

        answer_file_fn = f"{args.answer_file_folder}/explanation.jsonl"

        print(f"Writing to {answer_file_fn}")
        with open(answer_file_fn, "w") as f:
            for i, state in enumerate(states):
                output_dict = {
                    "image_1": arguments[i]["image_1"],
                    "image_2": arguments[i]["image_2"],
                    "class_1": arguments[i]["class_1"],
                    "class_2": arguments[i]["class_2"],
                    "question": arguments[i]["question"],
                    "answer": state,
                }
                f.write(json.dumps(output_dict) + "\n")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-file-folder", type=str, default="sglang_explanations")
    parser.add_argument("--image-folder", type=str, default="./train")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-per-pair", type=int, default=30)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = add_common_sglang_args_and_parse(parser)
    main(args)