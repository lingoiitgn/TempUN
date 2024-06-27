import os
import argparse
import time

parser = argparse.ArgumentParser(description='Temporal Inference')
parser.add_argument('--cuda', type=str, default="0", help='Cuda device')
parser.add_argument('--model', type=str, required=True, help='Model name') 
parser.add_argument('--hub-model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model name')
parser.add_argument('--base-dir', type=str, default='/path_to_csvs', help='Path to the csvs') 
parser.add_argument('--out-dir', type=str, default='/path_to_results', help='Path to the output directory')
parser.add_argument('--year', type=int, default=1947, help='Year to run inference')

args = parser.parse_args()

for year in range(1947, 2023):
    model_name = args.hub_model_name.split("/")[0]
    if year < args.year:
        continue
    print(f"Year: {year}")

    # Run inference
    print(f"Running inference for {year}")
    os.system(f"CUDA_VISIBLE_DEVICES={args.cuda} python /path_to_code4.py/code4.py --model {args.model} --base-dir {args.base_dir} --out-dir {args.out_dir}/{model_name}/{year} --year {year} --hub-model-name {args.hub_model_name}")

    time.sleep(3)
