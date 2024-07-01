# TempUN Dataset
We introduce **TempUN**, an extensive and balanced dataset ranging from 10,000 BCE to 2100 CE. The dataset was obtained through web scraping from the following [Our World in Data](https://ourworldindata.org/) is curated based on global issues and priorities as delineated by the [United Nations](https://www.un.org/en/global-issues) and [EU](https://www.undp.org/european-union/our-focus). **TempUN** mitigates the bias towards immutable facts found in TempLAMA; only 16.13\% of its facts remain unaltered, resulting in 83.87\% of the facts being subject to change. **TempUN** encompasses approximately 462,894 records, from which a substantial number of temporal prompts, 9,497,502 (denoted as **Large**), have been derived. However, to accommodate the computational constraints of larger models, we also offer a compressed version of the dataset, which consists of a random selection of 1,907 instances corresponding to 104,130 prompts (referred to as **Small**).

Dataset Link: [Drive](https://drive.google.com/drive/u/2/folders/1ci_Ni4ab5fQ5-x4Ly9n_FW9_rlBOKqiM)

# Inference

## Zeroshot
### Open-source models:
#### Using vLLM:
We have utillized vLLM for getting inference from `phi-2`, `mistral-instruct-v0.2`, and `llama-2-chat`. The code for inference is located in the `inference/zeroshot/open-source/inf_vllm.py` file. To get inference run below command:

    CUDA_VISIBLE_DEVICES=0 python inf_vllm.py --hub-model-name mistralai/Mistral-7B-Instruct-v0.2 --base-dir /path_to_csvs --out-dir /path_to_results --year 1947

#### Using transformers:
We have utillized transformers for getting inference from `flan-t5-xl`, and `phi-3-medium-instruct`. The code for inference is located in the `inference/zeroshot/open-source/inf_transformers.py` file. To get inference run below command:

    CUDA_VISIBLE_DEVICES=0 python inf_transformers.py --hub-model-name microsoft/Phi-3-medium-128k-instruct --base-dir /path_to_csvs --out-dir /path_to_results --year 1947

#### Using Groq:
We have utillized transformers for getting inference from `gemma-1.1-7b-it`, `Meta-Llama-3-8B-Instruct`, `Mixtral-8x7B-Instruct-v0.1`, and `Meta-Llama-3-70B-Instruct`. The code for inference is located in the `inference/zeroshot/open-source/inf_groq.py` file. To get inference run below command:

    python inf_groq.py --base-dir /path_to_csvs --out-dir /path_to_results --year 1947

- Change the model in the script to run the inference for the desired LLM.

### Closed-source models:
We have performed inference on three closed-source models: `gpt-3.5-turbo`, `gpt-4`, and `gemini-pro`. 

- The code for inference for models `gpt-3.5-turbo` and `gpt-4` is located in the `inference/zeroshot/closed-source/inf_openai.py` file. To get inference run below command:

      python inf_openai.py --api-key YOUR_OPENAI_API_KEY --api-base YOUR_API_BASE_URL --engine YOUR_ENGINE_NAME --base-dir /path_to_csvs --out-dir /path_to_results --year 1947

- The code for inference for model `gemini-pro` is located in the `inference/zeroshot/closed-source/inf_gemini.py` file. To get inference run below command:

      python inf_gemini.py --api-key YOUR_GEMINI_API_KEY --base-dir /path_to_csvs --out-dir /path_to_results --year 1947

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

\begin{table*}[!tbh]
\centering
\begin{tabular}{lcccccccc}  \hline
\textbf{Models} & \textbf{Generation} & \textbf{$DB$} & \textbf{$CP$} & \textbf{$WB$} & \textbf{$MM$} & \textbf{$RB$} & \textbf{$TB$} & \textbf{Average} \\  \hline
 & \textbf{C$\uparrow$} & .11 & 0 & .18 & .08 & .09 & .06 & .09 \\
 & \textbf{I$\downarrow$} & .89 & .97 & .82 & .92 & .89 & .93 & .90 \\
\multirow{-3}{*}{\texttt{phi-2}} & \textbf{N$\downarrow$} & \textbf{0} & .03 & \textbf{0} & \textbf{0} & .02 & .01 & .01 \\  \hline
 & \textbf{C$\uparrow$} & .38 & .40 & .20 & .24 & .20 & .03 & .30 \\
 & \textbf{I$\downarrow$} & .62 & .60 & .80 & .76 & .79 & .97 & .69 \\
\multirow{-3}{*}{\texttt{flan-t5-xl}} & \textbf{N$\downarrow$} & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} & .01 & \textbf{0} & \textbf{0} \\  \hline
 & \textbf{C$\uparrow$} & .37 & .43 & .20 & .23 & .34 & \textbf{.08} & .27 \\
 & \textbf{I$\downarrow$} & .51 & .57 & .80 & .64 & .66 & .71 & .65 \\
\multirow{-3}{*}{\texttt{mistral-instruct}} & \textbf{N$\downarrow$} & .12 & \textbf{0} & \textbf{0} & .13 & \textbf{0} & .22 & .08 \\  \hline
 & \textbf{C$\uparrow$} & .21 & .45 & .22 & .15 & .22 & .05 & .21 \\
 & \textbf{I$\downarrow$} & .76 & .55 & .78 & .81 & .79 & .93 & .77 \\
\multirow{-3}{*}{\texttt{llama-2-chat}} & \textbf{N$\downarrow$} & .03 & \textbf{0} & \textbf{0} & .04 & \textbf{0} & .02 & .02 \\  \hline
 & \textbf{C$\uparrow$} & .21 & .42 & .15 & .12 & .14 & .03 & .19 \\
 & \textbf{I$\downarrow$} & .77 & .58 & .85 & .88 & .86 & .94 & .79 \\  
\multirow{-3}{*}{\texttt{gemma-7b-it}} & \textbf{N$\downarrow$} & .02 & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} & .03 & .01 \\ \hline
 & \textbf{C$\uparrow$} & .39 & .39 & .19 & .18 & .24 & .07 & .31 \\
 & \textbf{I$\downarrow$} & .61 & .61 & .81 & .82 & .76 & .93 & .69 \\  
\multirow{-3}{*}{\texttt{llama-3-8b}} & \textbf{N$\downarrow$} & .01 & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} \\\hline
 & \textbf{C$\uparrow$} & .09 & \textbf{.49} & .37 & .10 & .01 & .01 & .14 \\
 & \textbf{I$\downarrow$} & \textbf{.16} & .47 & \textbf{.31} & \textbf{.27} & \textbf{.03} & .53 & \textbf{.24} \\
\multirow{-3}{*}{\texttt{phi-3-medium}} & \textbf{N$\downarrow$} & .74 & .05 & .33 & .63 & .96 & .46 & .62 \\  \hline
 & \textbf{C$\uparrow$} & .33 & .34 & .29 & .18 & .29 & .03 & .28 \\
 & \textbf{I$\downarrow$} & .61 & .64 & .71 & .82 & .71 & .94 & .68 \\
\multirow{-3}{*}{\texttt{mixtral-8x7b}} & \textbf{N$\downarrow$} & .07 & .02 & \textbf{0} & \textbf{0} & \textbf{0} & .03 & .04 \\  \hline
 & \textbf{C$\uparrow$} & \textbf{.40} & .37 & \textbf{.55} & \textbf{.37} & \textbf{.38} & .01 & \textbf{.37} \\
 & \textbf{I$\downarrow$} & .60 & .63 & .45 & .63 & .62 & .99 & .63 \\
\multirow{-3}{*}{\texttt{llama-3-70b}} & \textbf{N$\downarrow$} & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} & \textbf{0} \\  \hline
{\color[HTML]{656565} } & {\color[HTML]{656565} \textbf{C$\uparrow$}} & {\color[HTML]{656565} .27} & {\color[HTML]{656565} .39} & {\color[HTML]{656565} .16} & {\color[HTML]{656565} .19} & {\color[HTML]{656565} .12} & {\color[HTML]{656565} 0} & {\color[HTML]{656565} .19} \\
{\color[HTML]{656565} } & {\color[HTML]{656565} \textbf{I$\downarrow$}} & {\color[HTML]{656565} .72} & {\color[HTML]{656565} .61} & {\color[HTML]{656565} .84} & {\color[HTML]{656565} .81} & {\color[HTML]{656565} .88} & {\color[HTML]{656565} .99} & {\color[HTML]{656565} .81} \\
\multirow{-3}{*}{{\color[HTML]{656565} \texttt{gpt-3.5-turbo}}} & {\color[HTML]{656565} \textbf{N$\downarrow$}} & {\color[HTML]{656565} .01} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} .01} & {\color[HTML]{656565} .01} & {\color[HTML]{656565} .01} \\  \hline
{\color[HTML]{656565} } & {\color[HTML]{656565} \textbf{C$\uparrow$}} & {\color[HTML]{656565} .29} & {\color[HTML]{656565} .02} & {\color[HTML]{656565} 0} & {\color[HTML]{656565} .29} & {\color[HTML]{656565} 0} & {\color[HTML]{656565} .01} & {\color[HTML]{656565} .10} \\
{\color[HTML]{656565} } & {\color[HTML]{656565} \textbf{I$\downarrow$}} & {\color[HTML]{656565} .35} & {\color[HTML]{656565} .98} & {\color[HTML]{656565} 1.00} & {\color[HTML]{656565} .50} & {\color[HTML]{656565} 1.00} & {\color[HTML]{656565} \textbf{.12}} & {\color[HTML]{656565} .66} \\
\multirow{-3}{*}{{\color[HTML]{656565} \texttt{gpt-4}}} & {\color[HTML]{656565} \textbf{N$\downarrow$}} & {\color[HTML]{656565} .36} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} .21} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} .87} & {\color[HTML]{656565} .24} \\  \hline
{\color[HTML]{656565} } & {\color[HTML]{656565} \textbf{C$\uparrow$}} & {\color[HTML]{656565} .29} & {\color[HTML]{656565} .38} & {\color[HTML]{656565} .34} & {\color[HTML]{656565} .15} & {\color[HTML]{656565} 0} & {\color[HTML]{656565} 0} & {\color[HTML]{656565} .19} \\
{\color[HTML]{656565} } & {\color[HTML]{656565} \textbf{I$\downarrow$}} & {\color[HTML]{656565} .71} & {\color[HTML]{656565} .62} & {\color[HTML]{656565} .66} & {\color[HTML]{656565} .85} & {\color[HTML]{656565} .99} & {\color[HTML]{656565} 1.00} & {\color[HTML]{656565} .80} \\
\multirow{-3}{*}{{\color[HTML]{656565} \texttt{gemini-pro}}} & {\color[HTML]{656565} \textbf{N$\downarrow$}} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} .01} & {\color[HTML]{656565} \textbf{0}} & {\color[HTML]{656565} \textbf{0}} \\   \hline
\end{tabular}%


# Fine-tuning
We have performed model fine-tuning in three different paradigms: **Yearwise Fine-Tuning (Y-FT)**, **Continual Learning (CL)** and **Random Fine-Tuning (R-FT)**. We have fine-tuned `phi-2`, `flan-t5-xl`, `mistral-instruct`, `llama-2-chat`, `gemma-7b-it`, `llama-3-8b`, and `phi-3-instruct` models on our **TempUN<sub>s</sub>** dataset.

More Details in the paper (Link Coming soon)
