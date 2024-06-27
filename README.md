# TempUN Dataset
We introduce **TempUN**, an extensive and balanced dataset ranging from 10,000 BCE to 2100 CE. The dataset was obtained through web scraping from the following [Our World in Data](https://ourworldindata.org/) is curated based on global issues and priorities as delineated by the [United Nations](https://www.un.org/en/global-issues) and [EU](https://www.undp.org/european-union/our-focus). **TempUN** mitigates the bias towards immutable facts found in TempLAMA; only 16.13\% of its facts remain unaltered, resulting in 83.87\% of the facts being subject to change. **TempUN** encompasses approximately 462,894 records, from which a substantial number of temporal prompts, 9,497,502 (denoted as **Large**), have been derived. However, to accommodate the computational constraints of larger models, we also offer a compressed version of the dataset, which consists of a random selection of 1,907 instances corresponding to 104,130 prompts (referred to as **Small**).

Dataset Link: [Drive](https://drive.google.com/drive/folders/1Qc-xqoWOACZI1uUxcKLsKuXoqHF6Sv5w?usp=sharing)

# Inference

## Zeroshot:

## After fine tuning:
We have performed model fine-tuning in three different paradigms:

1. Yearwise Fine-Tuning (Y-FT)
2. Continual Learning (CL)
3. Random Fine-Tuning (R-FT)

### For Y-FT and CL:
#### Using vLLM:
At the time of performing inference, from all fine-tuned models only `phi-2`, `mistral-instruct`, `llama-2-chat`, and `llama-3-8b` were supported via **vLLM** (a fast and easy-to-use library for LLM inference). The code for inference is located in the `inference/after_FT/inf_vllm` directory.

- The lora_to_original.py script is designed to merge a pre-trained base model with an adapter model (fine-tuned model) using Parameter-Efficient Fine-Tuning (PEFT) and then save the merged model.

Performing Inference:

    The code4.py script performs inference using the fine-tuned LLMs and saves the results. It reads prompts from CSV files, generates responses using the LLM, extracts specific answers from the responses, and saves the results to new CSV files.

Orchestrating the Process:

    The run.py script orchestrates the entire process. It loads different checkpoints of fine-tuned models, converts them using the lora_to_original.py script, and then performs inference using the code4.py script.


More Details in the paper (Link Coming soon)
