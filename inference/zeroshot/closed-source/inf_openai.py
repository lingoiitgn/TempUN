import os
import argparse
import pandas as pd
import openai

parser = argparse.ArgumentParser(description='Temporal Inference')
parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--api-type', type=str, default='azure', help='OpenAI API type (e.g., azure)')
parser.add_argument('--api-base', type=str, required=True, help='OpenAI API base URL')
parser.add_argument('--api-version', type=str, default='2023-07-01-preview', help='OpenAI API version')
parser.add_argument('--engine', type=str, required=True, help='OpenAI engine name (e.g., gpt-3.5-turbo, gpt-4)')
parser.add_argument('--base-dir', type=str, default='/path_to_csvs', help='Path to the csvs')
parser.add_argument('--out-dir', type=str, default='/path_to_results', help='Path to the output directory')
parser.add_argument('--year', type=int, default=1947, help='Year to run inference')

args = parser.parse_args()

# Configure the OpenAI API
openai.api_type = args.api_type
openai.api_base = args.api_base
openai.api_version = args.api_version
openai.api_key = args.api_key

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
                        prompt_id = prompts_df['id'][i]
                        message_text = [
                            {"role": "system", "content": "Only answer with the correct answer, not even is correct. Your Answer shouldn't have any text other than the correct option and its correct option alphabet inside parenthesis"},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": ""}
                        ]

                        try:
                            completion = openai.ChatCompletion.create(
                                engine=args.engine,
                                messages=message_text,
                                temperature=0.1,
                                max_tokens=40,
                                top_p=0.95,
                                frequency_penalty=0,
                                presence_penalty=0,
                                stop=None
                            )

                            generated_text = completion['choices'][0]['message']['content']
                            generated_answers.append(generated_text)
                            ids.append(prompt_id)
                            print(generated_text, prompt_id, i)

                        except Exception as e:
                            print(f"Error processing prompt: {prompt}, Error: {str(e)}")
                            generated_answers.append("")  # Append empty string for error cases
                            ids.append(prompt_id)

                        counter += 1

                        if counter % 10 == 0:
                            sleep(5)  # Sleep for 5 seconds after every 10 requests

                    save_dir = os.path.join(args.out_dir, str(year))
                    os.makedirs(save_dir, exist_ok=True)
                    file_name = f'results_{file}'
                    answers_df = pd.DataFrame({'id': ids, 'answer': generated_answers})
                    answers_df.to_csv(os.path.join(save_dir, file_name), index=False)

print("Inference completed.")

