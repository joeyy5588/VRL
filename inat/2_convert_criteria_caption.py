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
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))

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

                description = data["answer"].lower().replace("image 1", "class 1").replace("image 2", "class 2")
                class_1 = data["class_1"]
                class_2 = data["class_2"]

                system_prompt = "I have a series of descriptions that I would like to convert into classification questions. For each description, response in JSON format which includes a question and provide specific labels for Class 1 and Class 2, based on the key distinguishing feature mentioned in the description."
                system_prompt += "\nExample description: The most distinctive feature that can be used to distinguish class 1 and class 2 is the type of fungus present. class 1 has a bright yellow, fuzzy fungus with a round shape, while class 2 has bright yellow, delicate flower-like structures growing from a dark gray tree branch."
                system_prompt += "\nExample response: {\"question\": \"What type of fungus is present?\", \"class_1\": \"bright yellow, fuzzy fungus with a round shape\", \"class_2\": \"bright yellow, delicate flower-like structures growing from a dark gray tree branch\"}"

                question_prompt = f"Now, convert this description: {description}" + "Please follow the same JSON format for the response. Response:" 

                arguments.append({
                    "description": description,
                    "system_prompt": system_prompt,
                    "question_prompt": question_prompt,
                    "class_1": class_1,
                    "class_2": class_2
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
            left_bracket = json_text.find("{")
            right_bracket = json_text.find("}")
            json_text = json_text[left_bracket:right_bracket+1]
            try:
                pred = json.loads(json_text)
            except:
                print(f"Error parsing json: {json_text}")
                pred = None

            preds.append(pred)

        answer_file_fn = f"{args.answer_file_folder}/{cat}_questions.jsonl"
        print(f"Writing to {answer_file_fn}")
        with open(answer_file_fn, "w") as f:
            for i, pred in enumerate(preds):
                output_dict = {
                    "description": arguments[i]["description"],
                    "class_1": arguments[i]["class_1"],
                    "class_2": arguments[i]["class_2"],
                    "pred": pred
                }
                f.write(json.dumps(output_dict) + "\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation-folder", type=str, default="sglang_explanations")
    parser.add_argument("--answer-file-folder", type=str, default="sglang_questions")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = add_common_sglang_args_and_parse(parser)
    main(args)