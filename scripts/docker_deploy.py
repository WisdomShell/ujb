import argparse
import random
import re
import subprocess
import time
import psutil
import requests

def log(message):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")
    
def check_gpu_usage(total_gpu_count=8):
    running_gup_id, cmds = [], []
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            output = result.stdout
            gpu_infos = re.findall(r'\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+(\w+)\s+.*', output)
            if gpu_infos:
                gpu_count = len(gpu_infos)
                log(f"Detected {gpu_count} GPU processes.")
                for gpu_info in gpu_infos:
                    gpu_id, pid, process_name = gpu_info
                    if process_name == "G": continue
                    running_gup_id.append(int(gpu_id))
                    log(f"GPU {gpu_id}: running - PID: {pid}, name: {process_name}")
                    cmds.append(" ".join(psutil.Process(int(pid)).cmdline()))
            else:
                log("当前没有进程占用显卡。")
                log("All GPUs are free.")
        else:
            log("Failed to run 'nvidia-smi'. Please make sure NVIDIA driver and nvidia-smi tools are installed.")
    except FileNotFoundError:
        log("Failed to find 'nvidia-smi'. Please make sure NVIDIA driver and nvidia-smi tools are installed.")
    
    free_gup_id = [i for i in range(total_gpu_count) if i not in running_gup_id]
    return free_gup_id, cmds

def docker_deploy(model_name_or_path, run_id, free_gpu_ids):
    port = str(10000 + random.randint(0, 20000))
    tgi_api_url = f"http://127.0.0.1:{port}"

    gpus = f"device={','.join([str(_id) for _id in free_gpu_ids])}"
    # Use 'subprocess' to execute bash commands and get the docker container id
    cmd_docker = [
        "docker", "run", "-d", "--gpus", f'"{gpus}"', "--shm-size", "2g", "-p", 
        f"{port}:80", "--env", 'LOG_LEVEL="info,text_generation_router=debug"', "-v",
        f"{model_name_or_path}:/{run_id}",
        "zzr0/text-generation-inference:codeshell-1.1.1",
        "--model-id", f"/{run_id}", "--num-shard", str(len(free_gpu_ids)),
        "--max-input-length=4096", "--max-total-tokens=5120",
        "--max-stop-sequences=12",
        "--trust-remote-code"
    ]
    print(" ".join(cmd_docker))
    container_id = subprocess.run(cmd_docker, capture_output=True, text=True).stdout.strip()

    # 等待 docker 容器完全启动
    for _ in range(180):
        try:
            response = requests.post(
                f"{tgi_api_url}/generate",
                headers={"Content-Type": "application/json"},
                json={"inputs": "test", "parameters": {"max_new_tokens": 20}}
            )
            if response.status_code == 200:
                return True, tgi_api_url, container_id
            else:
                print("waiting for docker container to start...")
                time.sleep(2)
        except:
            print("waiting for docker container to start...")
            time.sleep(2)
    return False, tgi_api_url, container_id

def main(args):
    api_urls = []
    container_ids = []
    free_gpu_ids, cmds = check_gpu_usage()
    free_gpu_ids = free_gpu_ids[:args.max_gpu_nums]
    test_gpu_sizes = [s for s in [1, 2, 4, 8] if s >= args.min_share_size]
    for size in test_gpu_sizes:
        success, api_url, container_id = docker_deploy(args.model_name_or_path, args.run_id, free_gpu_ids[:size])
        container_ids.append(container_id)
        cmd_kill = ["docker", "kill", container_id]
        subprocess.run(cmd_kill)
        print("success", success)
        if success:
            break
    print("size", size)
    for i in range(len(free_gpu_ids) // size):
        gpu_ids = free_gpu_ids[size * i:size * (i + 1)]
        success, api_url, container_id = docker_deploy(args.model_name_or_path, args.run_id, gpu_ids)
        container_ids.append(container_id)
        if success:
            api_urls.append(api_url)
    
    with open(args.api_urls, 'w') as file:
        file.write(','.join(api_urls))
    
    with open(args.container_ids, 'w') as file:
        file.write(' '.join(container_ids))

def paras_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_urls", type=str, default="./tgi_api_urls.txt", help="TGI API urls file path")
    parser.add_argument("--container_ids", type=str, default="./tgi_container_ids.txt", help="TGI container ids file path")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models", required=True)
    parser.add_argument("--run_id", type=str, help="Save run id", required=True)
    parser.add_argument("--max_gpu_nums", type=int, default=4, help="Max gpu nums to deploy TGI.")
    parser.add_argument("--min_share_size", type=int, default=1, help="How many gpus share one model.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args =  paras_args()
    main(args)
    