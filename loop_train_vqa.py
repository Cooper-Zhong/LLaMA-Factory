import subprocess
import random
import os
import argparse
from multiprocessing import Process, Queue


gpu_pool = [2,3]
# total number of tasks to train for each dataset
experiment_count = 4


CLEAN_DATA_NAMES = ["vqav2_single_1_clean", "vqav2_multi_2_clean", "vqav2_multi_3_clean"]

BACKDOOR_DATA_NAMES = [
            # "vqav2_backdoor_multi_2_rate_0_1", 
            # "vqav2_backdoor_multi_2_rate_0_2", 
            # "vqav2_backdoor_multi_3_rate_0_1", 
            # "vqav2_backdoor_multi_3_rate_0_2", 
            # "vqav2_backdoor_multi_2_rate_0_1_unified_pos_1", 
            # "vqav2_backdoor_multi_2_rate_0_1_unified_pos_2", 
            # "vqav2_backdoor_multi_2_rate_0_1_unified_pos_3", 
            # "vqav2_backdoor_multi_2_rate_0_1_unified_pos_4", 
            # "vqav2_backdoor_multi_3_rate_0_1_unified_pos_1",
            # "vqav2_backdoor_multi_3_rate_0_1_unified_pos_2",
            # "vqav2_backdoor_multi_3_rate_0_1_unified_pos_2",
            # "vqav2_backdoor_multi_3_rate_0_1_unified_pos_4",
            # "vqav2_backdoor_multi_2_rate_0_2_unified_pos_1",
            # "vqav2_backdoor_multi_2_rate_0_2_unified_pos_2",
            # "vqav2_backdoor_multi_2_rate_0_2_unified_pos_3",
            # "vqav2_backdoor_multi_2_rate_0_2_unified_pos_4",
            # "vqav2_backdoor_multi_3_rate_0_2_unified_pos_1",
            # "vqav2_backdoor_multi_3_rate_0_2_unified_pos_2",
            # "vqav2_backdoor_multi_3_rate_0_2_unified_pos_3",
            # "vqav2_backdoor_multi_3_rate_0_2_unified_pos_4",
            'vqav2_train_data_adj_shuffled',
            'vqav2_train_data_adj_sequential'
        ]

# base_model_id = "/data1/models/Qwen2-VL-2B-Instruct"
base_model_id = "/data1/zzy/Qwen2-VL-7B-Instruct"

# finetuned_models_directory = "/data1/zzy/backdoor_result/Qwenvl2_vqav2"
finetuned_models_directory = '/data_sda/zhiyuan/backdoor_result/Qwenvl2_vqav2'

# Check if output directory exists
def check_output_dir_exists(task: dict):
    id = task['task_id']
    model_prefix = task['model_prefix']
    output_dir = f"{finetuned_models_directory}/{model_prefix}_{id}"
    adapter_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    return os.path.exists(output_dir) and os.path.isfile(adapter_model_path)

# Run training on GPU
def run_training_on_gpu(gpu_id: int, task_queue: Queue):
    while not task_queue.empty():
        task = task_queue.get()
        task_id = task['task_id']
        model_prefix = task['model_prefix']
        data_name = task['data_name']
        num_train_epochs = task['num_train_epochs']
        learning_rate = task['learning_rate']
        gradient_accumulation_steps = task['gradient_accumulation_steps']
        per_device_train_batch_size = task['per_device_train_batch_size']

        output_dir = f"{finetuned_models_directory}/{model_prefix}_{task_id}"
        if check_output_dir_exists(task):
            print(f"GPU {gpu_id}: Skipping task {task_id}, model {model_prefix} already exists")
            continue

        command = f"CUDA_VISIBLE_DEVICES={gpu_pool[gpu_id]} python src/train.py " \
                  f"--stage sft " \
                  f"--do_train " \
                  f"--model_name_or_path {base_model_id} " \
                  f"--dataset {data_name} " \
                  f"--template qwen2_vl " \
                  f"--finetuning_type lora " \
                  f"--lora_target all " \
                  f"--lora_rank 16 " \
                  f"--output_dir {output_dir} " \
                  f"--overwrite_cache " \
                  f"--preprocessing_num_workers 16 " \
                  f"--per_device_train_batch_size {per_device_train_batch_size} " \
                  f"--gradient_accumulation_steps {gradient_accumulation_steps} " \
                  f"--lr_scheduler_type cosine " \
                  f"--logging_steps 10 " \
                  f"--overwrite_output_dir " \
                  f"--save_total_limit 1 " \
                  f"--learning_rate {learning_rate} " \
                  f"--num_train_epochs {num_train_epochs} " \
                  f"--plot_loss " \
                  f"--bf16"

        print(f"GPU {gpu_id}: Running training for task {task_id} with epochs={num_train_epochs}, lr={learning_rate}, grad_acc_steps={gradient_accumulation_steps}")
        subprocess.run(command, shell=True)

# Distribute tasks
def assign_training(experiment_count: int, goal: str, gpu_count: int):
    # specify the model prefix and data name here
    if goal == "clean":
        data_names = CLEAN_DATA_NAMES
        model_prefixes = ['Qwenvl2-7b' + f'_{data}' for data in data_names]
    elif goal == "backdoor":
        data_names = BACKDOOR_DATA_NAMES
        # TODO: change name format for backdoor models
        model_prefixes = ['Qwenvl2-7b' + f'_{data}' for data in data_names]

    all_tasks = []
    for i, model_prefix in enumerate(model_prefixes):
        data = data_names[i]
        for j in range(experiment_count):
            num_train_epochs = random.choice([1.0, 2.0, 3.0])
            learning_rate = random.choice([1e-5, 2e-5, 3e-5, 5e-5, 1e-4])
            gradient_accumulation_steps = random.choice([2, 4, 8])
            per_device_train_batch_size = random.choice([1,2,4])
            task = {
                'task_id': j,
                'model_prefix': model_prefix,
                'data_name': data,
                'num_train_epochs': num_train_epochs,
                'learning_rate': learning_rate,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'per_device_train_batch_size': per_device_train_batch_size
            }
            all_tasks.append(task)

    tasks_to_run = [task for task in all_tasks if not check_output_dir_exists(task)]
    task_queue = Queue()
    for task in tasks_to_run:
        task_queue.put(task)
    
    processes = []
    for gpu_id in range(gpu_count):
        p = Process(target=run_training_on_gpu, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, required=True, help="clean or backdoor")
    parser.add_argument("--experiment_count", type=int, default=4, help="Number of experiments for each dataset")
    experiment_count = parser.parse_args().experiment_count
    goal = parser.parse_args().goal
    assign_training(experiment_count, goal, len(gpu_pool))