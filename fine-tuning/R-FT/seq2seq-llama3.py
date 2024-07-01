from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig
from trl import SFTTrainer
import os
import argparse
from rich.pretty import pprint
import pandas as pd

parser = argparse.ArgumentParser(description='Finetune LLaMA')
parser.add_argument('--dataset-path', type=str, default="/path_to_csvs", help='train dataset: Folder path containing csvs')
parser.add_argument('--start-year', type=int, default=1947, help='start year')
parser.add_argument('--end-year', type=int, default=2020, help='end year')
parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B', help='model name')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--batch-size', type=int, default=32, help='train batch size')
parser.add_argument('--num', type=bool, default=False, help='numerical or non-numerical')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--save-limit', type=int, default=1, help='save limit for checkpoints')
parser.add_argument('--patience', type=int, default=3, help='Early Stopping patience')
parser.add_argument('--prefix', type=str, default='', help='prefix')

args = parser.parse_args()
pprint(args)


DATASET_FOLDER = args.dataset_path
MODEL_OUTPUT_FOLDER = f"models/llama3/{args.start_year}-{args.end_year}"
os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)




################### Loading Dataset ###################
csvs = os.listdir(args.dataset_path)
csvs = [x for x in csvs if x.endswith(".csv")]
csvs = sorted(csvs)
if "yearly_freq.csv" in csvs:
    csvs.remove("yearly_freq.csv")

data = pd.DataFrame()
for csv in csvs:
    name = csv.split(".")[0]
    if args.start_year <= int(name) <= args.end_year:
        print("Loading", csv)
        temp = pd.read_csv(os.path.join(args.dataset_path, csv))
        data = pd.concat([data, temp], ignore_index=True)

data = data.astype(str)
data["text"] = data["query"] + " " + data["answer"]

if "year" in data.columns:
    data.drop(columns=["year"], inplace=True)
if "frequency" in data.columns:
    data.drop(columns=["frequency"], inplace=True)

data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
# eval_data = data.sample(frac=0.2, random_state=args.seed).reset_index(drop=True)
eval_data = data

dataset = Dataset.from_pandas(data)
eval_dataset = Dataset.from_pandas(eval_data)

# Getting the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

# Preprocessing the dataset(generate the tokens and remove remaining columns)
def preprocess_function(examples):
    return tokenizer([examples['text'][i] for i in range(len(examples['id']))])

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names,
)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=False,
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    bias="none",
    lora_dropout=0.05, # Conventional
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(args.model_name, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

training_arguments = TrainingArguments(
    output_dir= MODEL_OUTPUT_FOLDER,
    num_train_epochs= args.epochs,
    per_device_train_batch_size= args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps= 1,
    optim="paged_adamw_32bit",
    logging_steps=100,
    logging_strategy="steps",
    learning_rate= args.lr,
    fp16= False,
    bf16= False,
    group_by_length= True,
    disable_tqdm=False,
    save_total_limit= args.save_limit,
    evaluation_strategy="epoch",
    report_to="tensorboard"

)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

print("Saving to", MODEL_OUTPUT_FOLDER)
trainer.train()
trainer.save_model(MODEL_OUTPUT_FOLDER+"/checkpoint")
