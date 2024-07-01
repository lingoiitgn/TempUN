import os 
import time

start_year = 1947
end_year = 2022
dataset_path = "path_to_yearly_dataset"
model_name = "microsoft/phi-2"
prev = None
seed = 1
batch_size = 8
numerical = "numerical"
epoch = 25
learning_rate = 2e-5
save_limit = 1
patience = 4
prefix = ""
cuda = "2"
logs_dir = "./logs/continual_phi2-micro"

os.makedirs(logs_dir, exist_ok=True)
for year in range(start_year, end_year+1):
    print(f"Year: {year}")
    if year != start_year and prev != None:
        model_name = prev

    command = f"CUDA_VISIBLE_DEVICES={cuda} python /path_to_seq2seq-phi2.py --dataset-path {dataset_path} --start-year {year} --end-year {year} --model-name {model_name} --seed {seed} --batch-size {batch_size} --num {numerical} --epochs {epoch} --lr {learning_rate} --save-limit {save_limit} --patience {patience} --prefix \"{prefix}\" | tee {logs_dir}/{year}.txt"
    print(command)
    os.system(command)

    # Set prev model
    with open(f"{logs_dir}/{year}.txt", "r") as f:
        for line in f.readlines():
            if line.find("Saving to") != -1:
                prev_chkpt = line.split("Saving to ")[1].split("\n")[0]
                dirs = os.listdir(prev_chkpt)
                if len(dirs) == 0:
                    print("No checkpoint found")
                    exit()
                dirs.sort(reverse=True) # The last checkpoint is the best one
                for d in dirs:
                    if d.find("checkpoint") != -1:
                        prev = f"{prev_chkpt}/{d}"
                        break
    print("="*100)
    time.sleep(5)