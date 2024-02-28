# modified from MT-Bench
import argparse
import json
import os
import random
import time
import concurrent.futures

import tqdm

from code_ujb.common import (
    reorg_output_file,
    chat_compeletion_tgi,
    chat_compeletion_openai,
)
from fastchat.model.model_adapter import get_conversation_template
from code_ujb import tasks

def run_generate(
    model_id,
    save_id,
    bench_name,
    gen_mode,
    question_begin,
    question_end,
    save_generations_path,
    temperature,
    max_new_tokens,
    num_samples,
    parallel
):
    task_bench = tasks.get_task(bench_name)
    os.makedirs(os.path.dirname(save_generations_path), exist_ok=True)
    with open(save_generations_path + ".tmp", "w") as fout:
        pass
    
    stop_str_list = []
    if gen_mode == "chat":
        conv = get_conversation_template(model_id)
        print(f"Using chat mode, and the conversation template is '{conv.name}'.")

        if isinstance(conv.stop_str, str):
            stop_str_list.append(conv.stop_str)
        elif isinstance(conv.stop_str, list):
            stop_str_list = conv.stop_str
        
    n_copy = num_samples
    all_tasks = []
    for i in range(len(task_bench.get_dataset())):
        if question_begin>=0: 
            if i < question_begin: continue
        if question_end>=0: 
            if i >= question_end: continue
        # Copy the question `n_copy` times
        for _ in range(n_copy):
            all_tasks.append({
                "task_idx": i,
                "task_id": task_bench.get_id_byidx(i),
                "model_id": model_id,
                "question": task_bench.get_prompt_byidx(i, mode=gen_mode),
                "stream_stop": task_bench.get_stream_stop(i, mode=gen_mode)
            })
    
    random.shuffle(all_tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for task in all_tasks:
            future = executor.submit(
                get_answer,
                task,
                gen_mode,
                model_id,
                max_new_tokens,
                temperature,
                stop_str_list,
                save_generations_path,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

def get_answer(
    task: dict, gen_mode: str, model: str, 
    max_new_tokens: int, temperature: float, stop_str_list: list,
    save_generations_path: str,
):
    temperature = args.temperature
    chat_state = None  # for palm-2 model
        
    conv = get_conversation_template(model)

    conv.append_message(conv.roles[0], task["question"])
    conv.append_message(conv.roles[1], None)

    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314",
                   "claude-v1", "claude-instant-v1", "claude-instant-1", "claude-2",
                   "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k"]:
        output = chat_compeletion_openai(model, conv, temperature, max_new_tokens)
    else:
        output = chat_compeletion_tgi(gen_mode, task, conv, temperature, max_new_tokens, stop_str_list)
        
    # Dump answers
    ans = {
        "task_idx": task["task_idx"], 
        "task_id": task["task_id"], 
        "outputs": [output],
        "gen_mode": gen_mode,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "tstamp": time.time(),
        "model_id": model,
    }

    with open(os.path.expanduser(save_generations_path + ".tmp"), "a") as fout:
        fout.write(json.dumps(ans) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--gen-mode",
        type=str,
        choices=["complete", "chat"],
        default="complete",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--save-generations-path", type=str, help="The output answer file.")
    parser.add_argument("--model-id", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument("--temperature", type=float, help="temperature.", default=0.2)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
        default=-1
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions.",
        default=-1
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    args = parser.parse_args()

    print(f"Evaluating Model ID: {args.model_path}")
    run_generate(
        model_id=args.model_path,
        save_id=args.model_id,
        bench_name=args.bench_name,
        gen_mode=args.gen_mode,
        question_begin=args.question_begin,
        question_end=args.question_end,
        save_generations_path=args.save_generations_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        parallel=args.parallel,
    )

    if args.save_generations_path:
        reorg_output_file(args.save_generations_path, args.num_samples)
