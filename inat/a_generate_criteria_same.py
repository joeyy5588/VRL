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
fungi_list = ["Rusavskia_elegans", "Teloschistes_chrysophthalmus", "Teloschistes_exilis", "Teloschistes_flavicans", "Xanthomendoza_fallax", "Xanthoria_parietina"]
fish_list = ["Thalassoma_bifasciatum", "Thalassoma_hardwicke", "Thalassoma_lucasanum", "Thalassoma_lunare", "Thalassoma_pavo"]
grass_list = ["Elymus_virginicus", "Elymus_canadensis", "Elymus_hystrix", "Elymus_elymoides", "Elymus_repens"]
berry_list = ["Arctostaphylos_patula", "Arctostaphylos_uva-ursi", "Arctostaphylos_pungens", "Arctostaphylos_glauca", "Arctostaphylos_nevadensis"]
herb_list = ["Scirpus_atrovirens", "Scirpus_microcarpus", "Scirpus_sylvaticus", "Scirpus_cyperinus", "Scirpus_pendulus"]

category_list = ["fungi", "fish", "grass", "berry", "herb"]
super_class_list = [fungi_list, fish_list, grass_list, berry_list, herb_list]
combinations_list = [list(combinations(super_class, 2)) for super_class in super_class_list]

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

    for cat, super_class in zip(category_list, super_class_list):
        print(f"Processing {cat}")
        arguments = []
        metadata_list = []
        for c in super_class:
            class_name = c
            for subdir in os.listdir(args.image_folder):
                if class_name in subdir:
                    class1_img_folder = os.path.join(args.image_folder, subdir)

            for num in range(args.sample_per_pair):
                img_fn_list = random.sample(os.listdir(class1_img_folder), 2)
                image_1 = os.path.abspath(os.path.join(class1_img_folder, img_fn_list[0]))
                image_2 = os.path.abspath(os.path.join(class1_img_folder, img_fn_list[1]))
                arguments.append({
                    "image_1": image_1,
                    "image_2": image_2,
                    "question": f"List the key features that not only shared by the {cat} in both images but also make this species distinct from other {cat}. Focus on unique or specific characteristics, such as detailed patterns in the arrangement, textures, color variations, or specific forms of growth on surfaces. Provide each feature as a distinct bullet point, capturing the essence of what makes this species visually identifiable.",
                    "class_name": class_name,
                })

                
        states = [None] * len(arguments)

        client = create_openai_client("http://127.0.0.1:30000/v1")

        for i in tqdm(range(len(arguments))):
            generation_success = False
            while not generation_success:
                state = multi_image_stream_request_test(client, arguments[i]["image_1"], arguments[i]["image_2"], arguments[i]["question"])
                if "-" in state:
                    generation_success = True

            states[i] = state

        answer_file_fn = f"{args.answer_file_folder}/{cat}_explanation.jsonl"

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