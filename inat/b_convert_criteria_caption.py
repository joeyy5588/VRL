import json
import os

from inat_utils import category_list, super_class_list
import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
import argparse
@sgl.function
def convert_qa(s, system_prompt, question_prompt, **kwargs):
    s += sgl.system(system_prompt)
    s += sgl.user(question_prompt)
    s += sgl.assistant(sgl.gen("answer", max_tokens=512))

def main(args):
    if not os.path.exists(args.answer_file_folder):
        os.makedirs(args.answer_file_folder)

    for cat, super_class in zip(category_list, super_class_list):
        print(f"Processing {cat}")

        explanation_file = f"{args.explanation_folder}/{cat}_explanation.jsonl"
        arguments = []
        with open(explanation_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                data = json.loads(line)

                description = data["answer"]
                class_name = data["class_name"]

                system_prompt = "I have a series of descriptions that I would like to convert into a list of structured sentences, where each item describes one specific feature of the species. For each description, response in a list format."
                system_prompt += "\nExample description: The berry in both images exhibits several distinctive characteristics that set it apart from other berry species:\n\n- **Flower Structure**: The flowers are small, with five petals each, and they form in clusters. The petals are delicate and appear to be a soft pink or white color.\n- **Leaf Arrangement**: The leaves are arranged in an opposite or alternate pattern, with each leaf having a distinct shape that is often described as oval with a pointed tip.\n- **Leaf Texture**: The leaves have a velvety texture, which is unique to this species.\n- **Stem and Branches**: The stems and branches have small thorns or are spiny, which can be a defense mechanism against herbivores.\n- **Foliage Color**: The foliage is a vibrant green, indicating a healthy, thriving plant.\n- **Berries**: The berries are small, round, and appear to be a dark red or purple color, typical of many berry species.\n- **Growth Environment**: Both images show the plant growing in a rocky, perhaps alpine environment, which suggests it has adapted to grow in challenging conditions.\n- **Unique Shape**: The leaves and flowers have a unique shape, with the leaves having a slightly wavy edge and the flowers having a bell-shaped form."
                system_prompt += "\nExample response: [\"Its flowers are small, with five petals each, and they form in clusters. The petals are delicate and appear to be a soft pink or white color.\","
                system_prompt += "\"The leaves are arranged in an opposite or alternate pattern, with each leaf having a distinct shape that is often described as oval with a pointed tip.\","
                system_prompt += "\"The leaves have a velvety texture, which is unique to this species.\","
                system_prompt += "\"The stems and branches have small thorns or are spiny, which can be a defense mechanism against herbivores.\","
                system_prompt += "\"The foliage is a vibrant green, indicating a healthy, thriving plant.\","
                system_prompt += "\"The berries are small, round, and appear to be a dark red or purple color, typical of many berry species.\","
                system_prompt += "\"The plant growing in a rocky, perhaps alpine environment, which suggests it has adapted to grow in challenging conditions.\","
                system_prompt += "\"The leaves and flowers have a unique shape, with the leaves having a slightly wavy edge and the flowers having a bell-shaped form.\"]"


                question_prompt = f"Now, convert this description: {description}" + "Please follow the same format for the response. Response:" 

                arguments.append({
                    "description": description,
                    "system_prompt": system_prompt,
                    "question_prompt": question_prompt,
                    "class_name": class_name,
                })        

        states = [None] * len(arguments)

        # Select backend
        backend = select_sglang_backend(args)
        sgl.set_default_backend(backend)

        states = convert_qa.run_batch(
            arguments, temperature=0, num_threads=args.parallel, progress_bar=True
        )

        preds = []
        for state in states:
            # extract the json dict from the generated text
            json_text = state["answer"].strip()
            left_bracket = json_text.find("[")
            right_bracket = json_text.find("]")
            json_text = json_text[left_bracket:right_bracket+1]
            try:
                pred = json.loads(json_text)
            except:
                json_text = state["answer"].strip()
                if "]" in json_text:
                    json_text = json_text.replace("\"\"", "\"")
                else:
                    json_text += "\"]"
                    json_text = json_text.replace("\"\"", "\"")

                try:
                    pred = json.loads(json_text)
                except:
                    print(f"Error parsing json: {json_text}")
                    pred = [json_text]

            preds.append(pred)

        answer_file_fn = f"{args.answer_file_folder}/{cat}_questions.jsonl"
        print(f"Writing to {answer_file_fn}")
        with open(answer_file_fn, "w") as f:
            question_id = 0
            for i, pred in enumerate(preds):
                for sentence in pred:
                    output_dict = {
                        "description": arguments[i]["description"],
                        "class_name": arguments[i]["class_name"],
                        "pred": sentence,
                        "question_id": question_id,
                    }
                    question_id += 1
                    f.write(json.dumps(output_dict) + "\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation-folder", type=str, default="sglang_explanations_72b_same")
    parser.add_argument("--answer-file-folder", type=str, default="sglang_questions_72b_same")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = add_common_sglang_args_and_parse(parser)
    main(args)