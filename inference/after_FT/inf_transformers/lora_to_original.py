import os
os.environ['HF_TOKEN'] = ''   # add HF access token

from peft import PeftConfig, PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description='LoRA to Original')
parser.add_argument('--model', type=str, default='google/gemma-1.1-7b-it', help='Model name')
parser.add_argument('--adapter', type=str, required=True, help='Adapter model name')
parser.add_argument('--output', type=str, default='original_model', help='Output folder')
args = parser.parse_args()

base_model_name = args.model
adapter_model_name = args.adapter

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(base_model_name)
print("PEFTing model")
model = PeftModel.from_pretrained(model, adapter_model_name)

print("Merge and unload")
model = model.merge_and_unload()
print("Saving")
model.save_pretrained(f"{args.output}")
