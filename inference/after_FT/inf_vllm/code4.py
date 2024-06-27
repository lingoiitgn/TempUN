import argparse
import re

parser = argparse.ArgumentParser(description='Inferencing using vLLM')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model name')
parser.add_argument('--hub-model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model name')
parser.add_argument('--base-dir', type=str, default='/path_to_csvs', help='Path to the csvs')
parser.add_argument('--out-dir', type=str, default='/path_to_results', help='Path to the output directory')
parser.add_argument('--year', type=int, default=1947, help='Year from which the inference is to be done')

args = parser.parse_args()

# Defining the seed
SEED = 1

MODEL=args.model

FOLDER_PATH = args.base_dir
last_folder = FOLDER_PATH.split("/")[-1]
results_folder=args.out_dir
# Defining a prompt suffix
SUFFIX = '. Provide only the correct option, without explaination.'

import torch

torch.cuda.empty_cache()

option_regex = re.compile(r"\([a-zA-Z]\)")
yes_no_regex = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

## vLLM code
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, trust_remote_code=True, seed = SEED, dtype="float16", tokenizer=args.hub_model_name)
sampling_params = SamplingParams(n=1, temperature=0.01, top_p=0.94, top_k=30, max_tokens=60)
ignore_folders = [""]
import os
import pandas as pd

for i in os.listdir(FOLDER_PATH):
    if i in ignore_folders:
        continue
    for j in os.listdir(os.path.join(FOLDER_PATH, i)):
        llm.seed = SEED
        csv_path = os.path.join(FOLDER_PATH, i, j)
        # print(csv_path)
        if not str(csv_path).endswith(".csv") or str(args.year) not in csv_path:
            continue
        print(f"-------Processing file: {os.path.join(i,j)}-------")
        prompts_df = pd.read_csv(csv_path)
        prompts_df['modified_prompt'] = prompts_df['query'].apply(lambda x: x + SUFFIX)
        prompts = prompts_df['modified_prompt'].tolist()

        outputs = llm.generate(prompts, sampling_params)
        answers = [output.outputs[0].text for output in outputs]
        extracted_ans = []

        for ans in answers:
            option_match = option_regex.findall(ans)
            yes_no_match = yes_no_regex.findall(ans)
            if option_match:
                extracted_ans.append(option_match[0].strip("()"))
            elif yes_no_match:
                extracted_ans.append(yes_no_match[-1].capitalize())
            else:
                extracted_ans.append("")

        prompts_df['generated_text'] = answers
        prompts_df['extracted_answer'] = extracted_ans
        os.makedirs(os.path.join(results_folder, i), exist_ok=True)
        res_path = os.path.join(results_folder, i, j)
        print(f"Saving to {res_path}")
        prompts_df.to_csv(res_path)
