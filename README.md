# TempUN Dataset
We introduce **TempUN**, an extensive and balanced dataset ranging from 10,000 BCE to 2100 CE. The dataset was obtained through web scraping from the following [Our World in Data](https://ourworldindata.org/) is curated based on global issues and priorities as delineated by the [United Nations](https://www.un.org/en/global-issues) and [EU](https://www.undp.org/european-union/our-focus). **TempUN** mitigates the bias towards immutable facts found in TempLAMA; only 16.13\% of its facts remain unaltered, resulting in 83.87\% of the facts being subject to change. **TempUN** encompasses approximately 462,894 records, from which a substantial number of temporal prompts, 9,497,502 (denoted as **Large**), have been derived. However, to accommodate the computational constraints of larger models, we also offer a compressed version of the dataset, which consists of a random selection of 1,907 instances corresponding to 104,130 prompts (referred to as **Small**).

Dataset Link: [Drive](https://drive.google.com/drive/folders/1Qc-xqoWOACZI1uUxcKLsKuXoqHF6Sv5w?usp=sharing)

# Inference

## Zeroshot
### Open-source models:
#### Using vLLM:
We have utillized vLLM for getting inference from `phi-2`, `mistral-instruct-v0.2`, and `llama-2-chat`. The code for inference is located in the `inference/zeroshot/inf_vllm.py` file. To get inference run below command:

    python inf_vllm.py --hub-model-name mistralai/Mistral-7B-Instruct-v0.2 --base-dir /path_to_csvs --out-dir /path_to_results --year 1947

#### Using transformers:
flan-t5-xl
#### Using Groq:
gemma-1.1-7b-it, Meta-Llama-3-8B-Instruct, phi-3-medium-instruct, Mixtral-8x7B-Instruct-v0.1, Meta-Llama-3-70B-Instruct, gpt-3.5-turbo, gpt-4, and gemini-pro 
### Closed-source models:

## After fine tuning
We have performed model fine-tuning in three different paradigms:

1. Yearwise Fine-Tuning (Y-FT)
2. Continual Learning (CL)
3. Random Fine-Tuning (R-FT)

### For Y-FT and CL:
#### Using vLLM:
At the time of performing inference, from all fine-tuned models only `phi-2`, `mistral-instruct`, `llama-2-chat`, and `llama-3-8b` were supported via **vLLM** (a fast and easy-to-use library for LLM inference). The code for inference is located in the `inference/after_FT/inf_vllm` directory.

- The `lora_to_original.py` script is designed to merge a pre-trained base model with an adapter model (fine-tuned model) using Parameter-Efficient Fine-Tuning (PEFT) and then save the merged model.
- The `code4.py` script performs inference using the fine-tuned LLMs and saves the results. It reads prompts from CSV files, generates responses using the LLM, extracts specific answers from the responses, and saves the results to new CSV files.
- The `run.py` script orchestrates the entire process. It loads different checkpoints of fine-tuned models, converts them using the `lora_to_original.py` script, and then performs inference using the `code4.py` script.

To get inference results, follow these steps:
1. Modify the paths in the `lora_to_original.py`, `code4.py`, and `run.py` scripts to match your environment and change the `model_name` to get inference from desired llm.
2. Run the `run.py` script with the following command:

       python run.py --model /path_to_fine-tuned_models


#### Using transformers:
For models like `flan-t5-xl`, `gemma-7b-it`, and `phi-3-medium-instruct`, which are not supported by vLLM, we use the **transformers** library for inference. The code for inference is located in the `inference/after_FT/inf_transformers` directory. The `lora_to_original.py` and `run.py` remains the same except for the `code4.py` script, which uses transformers instead of vLLM.

- Steps to get inference remains the same as they were in vLLM section.

### For R-FT:
The `lora_to_original.py` and `code4.py` scripts remain the same (use `code4.py` from either the `inf_vllm` or `inf_transformers` folder based on the given model). However, instead of `run.py`, we use `run_full.py`, located at `inference/after_FT/run_full.py`.

To get inference results, follow these steps:
1. Modify the paths in the `lora_to_original.py`, `code4.py`, and `run_full.py` scripts to match your environment and change the `model_name` to get inference from desired llm.
2. Run the `lora_to_original.py` script with the following command:

       python lora_to_original.py --adapter /path_to_fine-tuned_model_checkpoint --output /path_to_save_model_and_use_this_path_for_run_full.py
3. Run the `run_full.py` script with the following command:

       python run_full.py --model /path_to_saved_model_from_step_2



More Details in the paper (Link Coming soon)
