import os
import argparse
import pandas as pd
import google.generativeai as genai
from time import sleep

parser = argparse.ArgumentParser(description='Temporal Inference')
parser.add_argument('--api-key', type=str, required=True, help='Gemini API key')
parser.add_argument('--base-dir', type=str, default='/path_to_csvs', help='Path to the csvs')
parser.add_argument('--out-dir', type=str, default='/path_to_results', help='Path to the output directory')
parser.add_argument('--year', type=int, default=1947, help='Year to run inference')

args = parser.parse_args()

# Configure the Gemini model
genai.configure(api_key=args.api_key)

generation_config = {
    "temperature": 0.01,
    "top_p": 0.95,
    "top_k": 30,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

REQUESTS_BEFORE_SLEEP = 10
ITERATIONS_BEFORE_APPEND = 5

for year in range(args.year, 2023):
    print(f"Year: {year}")

    for root, dirs, files in os.walk(args.base_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                if str(year) in csv_path:
                    print(f"-------Processing file: {csv_path}-------")
                    prompts_df = pd.read_csv(csv_path)
                    prompts_df['modified_prompt'] = prompts_df['query']  # Ensure 'modified_prompt' column exists
                    generated_answers = []
                    ids = []
                    counter = 0

                    for i in range(len(prompts_df)):
                        prompt = prompts_df['modified_prompt'][i]
                        prompt_parts = [prompt]  # Ensure prompt_parts is a list
                        prompt_id = prompts_df['id'][i]
                        response = model.generate_content(prompt_parts)
                        response.resolve()
                        ids.append(prompt_id)
                        if len(response.parts) == 0:
                            generated_answers.append('0')
                            print(response.candidates[0].content.parts, prompt_id, i)
                        else:
                            generated_answers.append(response.parts[0].text)
                            print(response.parts[0].text, prompt_id, i)

                        counter += 1

                        if counter >= ITERATIONS_BEFORE_APPEND:
                            answers_df = pd.DataFrame({'answer': generated_answers, 'id': ids})
                            save_dir = os.path.join(args.out_dir, str(year))
                            os.makedirs(save_dir, exist_ok=True)
                            file_path = os.path.join(save_dir, f'gemini_results_{file}')
                            if not os.path.exists(file_path):
                                answers_df.to_csv(file_path, index=False, mode='w')
                            else:
                                answers_df.to_csv(file_path, index=False, mode='a', header=False)

                            generated_answers = []
                            ids = []
                            counter = 0

                        if counter % REQUESTS_BEFORE_SLEEP == 0:
                            sleep(5)

                    if generated_answers:
                        answers_df = pd.DataFrame({'answer': generated_answers, 'id': ids})
                        save_dir = os.path.join(args.out_dir, str(year))
                        os.makedirs(save_dir, exist_ok=True)
                        file_path = os.path.join(save_dir, f'gemini_results_{file}')
                        if not os.path.exists(file_path):
                            answers_df.to_csv(file_path, index=False, mode='w')
                        else:
                            answers_df.to_csv(file_path, index=False, mode='a', header=False)

print("Inference completed.")
