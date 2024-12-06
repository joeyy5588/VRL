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

    arguments = []
    metadata_list = []
    for subdir in os.listdir(args.image_folder):
        class1_img_folder = os.path.join(args.image_folder, subdir)

        for num in range(args.sample_per_pair):
            img_fn_list = random.sample(os.listdir(class1_img_folder), 2)
            image_1 = os.path.abspath(os.path.join(class1_img_folder, img_fn_list[0]))
            image_2 = os.path.abspath(os.path.join(class1_img_folder, img_fn_list[1]))
            arguments.append({
                "image_1": image_1,
                "image_2": image_2,
                # "question": f"Describe the key features that consistently characterize this species across both images. Focus on identifying patterns, colors, textures, shapes, or other visual characteristics that are present in both images.",
                "question": f"List the key features that not only shared by the object in both images but also make this object distinct from other objects. Focus on unique or specific characteristics, such as detailed patterns in the arrangement, textures, color variations, or specific forms of growth on surfaces. Provide each feature as a distinct bullet point, capturing the essence of what makes this object visually identifiable.",
                "class_name": subdir,
            })

                
    states = [None] * len(arguments)

    # Select backend
    # backend = select_sglang_backend(args)
    # sgl.set_default_backend(backend)

    # states = image_difference_qa.run_batch(
    #     arguments, temperature=0, num_threads=args.parallel, progress_bar=True
    # )
    client = create_openai_client("http://127.0.0.1:30000/v1")

    for i in tqdm(range(len(arguments))):
        generation_success = False
        while not generation_success:
            state = multi_image_stream_request_test(client, arguments[i]["image_1"], arguments[i]["image_2"], arguments[i]["question"])
            if "-" in state:
                generation_success = True

        states[i] = state

    answer_file_fn = f"{args.answer_file_folder}/explanation.jsonl"

    print(f"Writing to {answer_file_fn}")
    with open(answer_file_fn, "w") as f:
        for i, state in enumerate(states):
            output_dict = {
                "image_1": arguments[i]["image_1"],
                "image_2": arguments[i]["image_2"],
                "class_name": arguments[i]["class_name"],
                "question": arguments[i]["question"],
                "answer": state,
            }
            f.write(json.dumps(output_dict) + "\n")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-file-folder", type=str, default="sglang_explanations_72b_same")
    parser.add_argument("--image-folder", type=str, default="./train")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-per-pair", type=int, default=10)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = add_common_sglang_args_and_parse(parser)
    main(args)