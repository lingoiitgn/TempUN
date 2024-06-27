import argparse
import os
import time

parser = argparse.ArgumentParser(description='Temporal Inference')
parser.add_argument('--cuda', type=str, default="0", help='Cuda device')
parser.add_argument('--models', type=str, required=True, help='Model name') 
parser.add_argument('--hub-model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model name')
parser.add_argument('--base-dir', type=str, default='/path_to_csvs', help='Path to the csvs') 
parser.add_argument('--out-dir', type=str, default='/path_to_results', help='Path to the output directory')
parser.add_argument('--year', type=int, default=1947, help='Year to run inference (Runs from this year)')

args = parser.parse_args()

years_folder = os.listdir(args.models)
years_folder.sort()
for year_ in years_folder:
    model_name = args.hub_model_name.split("/")[0]
    year = year_.split("-")[0]
    if int(year) < args.year:
        continue
    print(f"Year: {year}")
    chkpoints = os.listdir(f"{args.models}/{year_}")
    chkpoints.sort()
    checkpoint = chkpoints[0] if "checkpoint" in chkpoints[0] else chkpoints[1]
    checkpoint = f"{args.models}/{year_}/{checkpoint}"
    print(f"Checkpoint: {checkpoint}")
    # Convert to original model
    os.system(f"CUDA_VISIBLE_DEVICES={args.cuda} python /path_to_lora_to_original.py/lora_to_original.py --model {args.hub_model_name} --adapter {checkpoint} --output /path_for_model_output_from_lora_to_original/{model_name}")

    # Run inference
    print(f"Running inference for {year}")
    os.system(f"CUDA_VISIBLE_DEVICES={args.cuda} python /path_to_code4.py/code4.py --model /path_for_model_output_from_lora_to_original/{model_name} --base-dir {args.base_dir} --out-dir {args.out_dir}/{model_name}/{year} --year {year} --hub-model-name {args.hub_model_name}")

    time.sleep(3)
