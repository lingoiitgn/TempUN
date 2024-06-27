import os
os.environ['HF_TOKEN'] = ''     # add huggingface access token 

import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset


parser = argparse.ArgumentParser(description='Inferencing using vLLM')
parser.add_argument('--seed', type=int, default=1, help='Seed for reproducibility')
parser.add_argument('--model', type=str, default='google/gemma-1.1-7b-it', help='Model name')
parser.add_argument('--hub-model-name', type=str, default='google/gemma-1.1-7b-it', help='Model name')
parser.add_argument('--base-dir', type=str, default='/path_to_csvs', help='Path to the csvs')
parser.add_argument('--out-dir', type=str, default='/path_to_results', help='Path to the output directory')
parser.add_argument('--year', type=int, default=1947, help='Year for which the inference is to be done')

args = parser.parse_args()

# Defining the seed
SEED = 1

MODEL=args.model

FOLDER_PATH = args.base_dir
last_folder = FOLDER_PATH.split("/")[-1]
results_folder=args.out_dir

torch.cuda.empty_cache()

class QueryDataset(Dataset):
    def __init__(self, queries):
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]


tokenizer = AutoTokenizer.from_pretrained('google/gemma-1.1-7b-it')
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.float16)

model.eval()

def generate_text_batch(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
    outputs = model.generate(inputs, do_sample=True, max_new_tokens=100, temperature=0.2, top_p=0.94, num_return_sequences=1)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

def extract_option(result):
    if pd.isna(result):
        return None
    
    matches = re.findall(r'\(([a-d])\)|([a-d])\)|\b(yes|no)\b', result.lower())
    last_match = None
    
    for match in matches:
        if match[0]:  
            last_match = match[0][-1]
        elif match[1]:  
            last_match = match[1][-1]
        elif match[2]:  
            last_match = match[2].capitalize()
    return last_match

ignore_folders = [""]
import os
import pandas as pd

for i in os.listdir(FOLDER_PATH):
    if i in ignore_folders:
        continue
    for j in os.listdir(os.path.join(FOLDER_PATH, i)):
        csv_path = os.path.join(FOLDER_PATH, i, j)
        # print(csv_path)
        if not str(csv_path).endswith(".csv") or str(args.year) not in csv_path:
            continue
        print(f"-------Processing file: {os.path.join(i,j)}-------")
        prompts_df = pd.read_csv(csv_path)
                        # Prepare the DataLoader for batching
        queries = prompts_df['query'] + '. Provide only the correct option, without explanation.'
        dataset = QueryDataset(queries)
        dataloader = DataLoader(dataset, batch_size=50)  # Adjust batch size as needed

        generated_texts = []
        for batch in tqdm(dataloader, desc="Generating text", leave=False):
            batch_generated_texts = generate_text_batch(batch)
            generated_texts.extend(batch_generated_texts)
        
        prompts_df['generated_text'] = generated_texts

        prompts_df['extracted_option'] = prompts_df['generated_text'].apply(extract_option)

        os.makedirs(os.path.join(results_folder, i), exist_ok=True)
        res_path = os.path.join(results_folder, i, j)
        print(f"Saving to {res_path}")
        prompts_df.to_csv(res_path)
