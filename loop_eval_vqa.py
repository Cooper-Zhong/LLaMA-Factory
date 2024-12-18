import subprocess
import os
import argparse
from multiprocessing import Process, Queue, current_process

# GPU和模型存储设置
gpu_pool = [2,3]

# base_model_id = "/data1/models/Qwen2-VL-2B-Instruct"
base_model_id = "/data1/zzy/Qwen2-VL-7B-Instruct"

# clean_csv_path = '/data1/zzy/backdoor_result/backdoor_Qwenvl2_vqav2.csv'
# clean_csv_path = '/data1/zzy/backdoor_result/test1.csv'
# finetuned_models_directory = "/data1/zzy/backdoor_result/Qwenvl2_vqav2"
finetuned_models_directory = '/data_sda/zhiyuan/backdoor_result/Qwenvl2_vqav2'


def run_testing(gpu_id: int, task_queue: Queue):
    while not task_queue.empty():
        task = task_queue.get()
        model = task['model_name']
        test_file = task['test_file']
        csv_path = task['csv_path']
        goal = task['goal']
        command = f"CUDA_VISIBLE_DEVICES={gpu_pool[gpu_id]} python run_testing.py " \
                    f"--base_model_id {base_model_id} " \
                    f"--finetuned_models_directory {finetuned_models_directory} " \
                    f"--model_to_eval {model} " \
                    f"--test_file {test_file} " \
                    f"--goal {goal} " \
                    f"--csv_path {csv_path} "
        
        subprocess.run(command, shell=True)

def find_test_data(model_name: str):
    if 'single_1' in model_name:
        return '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_test_data_clean_single_1.json'
    elif 'multi_2' in model_name:
        return '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_test_data_clean_multi_2.json'
    elif 'multi_3' in model_name:
        return '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_test_data_clean_multi_3.json'
    elif 'adj' in model_name:
        return '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_test_data_adj.json'
    else:
        return None

# in order to see model's performance on training data
def find_train_data(model_name: str):
    if 'single_1' in model_name:
        return '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_train_data_clean_single_1.json'
    elif 'multi_2' in model_name:
        return '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_train_data_clean_multi_2.json'
    elif 'multi_3' in model_name:
        return '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_train_data_clean_multi_3.json'
    elif 'adj' in model_name:
        return '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_train_eval_data_adj.json'
    else:
        return None


def assign_testing(gpu_count: int, goal: str, csv_path: str):
    # specify the model prefix and test data
    single_1_test_data = '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_test_data_clean_single_1.json'
    multi_2_test_data = '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_test_data_clean_multi_2.json'
    multi_3_test_data = '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_test_data_clean_multi_3.json'
    adj_test_data = '/data1/zzy/datasets/backdoorvlm/vqav2/vqav2_test_data_adj.json'
    # TODO: update test data for unified trigger

    model_prefix_eval = [
        # "Qwenvl2-7b_vqav2_single_1_clean",
        # "Qwenvl2-7b_vqav2_multi_2_clean",
        # "Qwenvl2-7b_vqav2_multi_3_clean",
        # "Qwenvl2-7b_vqav2_backdoor_multi_2",
        # "Qwenvl2-7b_vqav2_backdoor_multi_3",
        "Qwenvl2-7b_vqav2_train_data_adj_sequential_4",
        # "Qwenvl2-7b_vqav2_train_data_adj_shuffled_2"
    ]

    all_models = os.listdir(finetuned_models_directory)
    all_models = [model for model in all_models if any([model.startswith(prefix) for prefix in model_prefix_eval])]
    if goal == 'clean':
        print('Evaluating clean models')
        all_models = [model for model in all_models if 'clean' in model]
    elif goal == 'backdoor':
        print('Evaluating backdoor models')
        all_models = [model for model in all_models if 'backdoor' in model]
    
    # Create a task queue and add all models to it
    task_queue = Queue()
    for model in all_models:
        length = len('Qwenvl2-7b_vqav2_backdoor_multi_3')
        # model_prefix = model[:length]
        task = {
            'model_name': model,
            # 'test_file': find_test_data(model),
            # to eval on training data
            'test_file': find_train_data(model),
            'csv_path': csv_path,
            'goal': goal
        }
        task_queue.put(task)

    processes = []
    
    for gpu_id in range(gpu_count):
        p = Process(target=run_testing, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def test(gpu_id, tasks_assigned):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_pool[gpu_id])
    print(os.environ['CUDA_VISIBLE_DEVICES'], tasks_assigned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, required=True, help="clean or backdoor")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the csv file to store the results")
    args = parser.parse_args()
    goal = args.goal
    csv_path = args.csv_path
    assign_testing(len(gpu_pool), goal, csv_path)