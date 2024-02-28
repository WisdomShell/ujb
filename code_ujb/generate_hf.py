# modified from MT-Bench
import argparse
import json
import os
import random
import time

import torch
from tqdm import tqdm

from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from code_ujb.common import reorg_output_file
from code_ujb import tasks

def run_generate(
    model_path,
    model_id,
    bench_name,
    gen_mode,
    question_begin,
    question_end,
    save_generations_path,
    temperature,
    max_input_tokens,
    max_new_tokens,
    num_samples,
    batch_size,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
):
    task_bench = tasks.get_task(bench_name)
    os.makedirs(os.path.dirname(save_generations_path), exist_ok=True)
    with open(save_generations_path + ".tmp", "w") as fout:
        pass
    if gen_mode == "chat":
        conv = get_conversation_template(model_id)
        print(f"Using chat mode, and the conversation template is '{conv.name}'.")
    
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    n_copy = num_samples // batch_size if num_samples % batch_size == 0 else num_samples // batch_size + 1
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
                "question": task_bench.get_prompt_byidx(i, mode=gen_mode),
            })
    
    num_world = num_gpus_total // num_gpus_per_model
    num_pad = (num_world - (len(all_tasks) % num_world)) % num_world
    random.shuffle(all_tasks)
    all_tasks.extend(all_tasks[:num_pad])
    chunk_size = len(all_tasks) // num_world
    ans_handles = []
    for i in range(0, len(all_tasks), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path=model_path,
                model_id=model_id,
                gen_mode=gen_mode,
                task_bench=task_bench,
                all_tasks=all_tasks[i : i + chunk_size],
                save_generations_path=save_generations_path,
                temperature=temperature,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                num_gpus_per_model=num_gpus_per_model,
                max_gpu_memory=max_gpu_memory,
                dtype=dtype,
            )
        )

    if use_ray:
        ray.get(ans_handles)

@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    gen_mode,
    task_bench,
    all_tasks,
    save_generations_path,
    temperature,
    max_input_tokens,
    max_new_tokens,
    batch_size,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
):
    save_generations_path = save_generations_path + ".tmp"
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    
    tasks_outputs = []
    for task in tqdm(all_tasks):
        torch.manual_seed(task["task_idx"])
        
        if gen_mode == "chat":
            conv = get_conversation_template(model_id)
            conv.append_message(conv.roles[0], task["question"])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        elif gen_mode == "complete":
            prompt = task["question"]
        else:
            raise NotImplementedError

        input_ids = tokenizer([prompt]).input_ids
        attention_mask = tokenizer([prompt]).attention_mask
        
        if len(input_ids[0]) > max_input_tokens:
            if gen_mode == "complete":
                input_ids[0] = input_ids[0][-max_input_tokens:]
                attention_mask[0] = attention_mask[0][-max_input_tokens:]
            else:
                input_ids[0] = input_ids[0][:max_input_tokens]
                attention_mask[0] = attention_mask[0][:max_input_tokens]

        if temperature < 1e-4:
            do_sample = False
        else:
            do_sample = True

        # some models may error out when generating long outputs
        try:
            output_ids_list = model.generate(
                input_ids=torch.as_tensor(input_ids).cuda(),
                attention_mask=torch.as_tensor(attention_mask).cuda(),
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=batch_size,
            )
            
            if gen_mode == "complete":
                outputs = tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)
            elif gen_mode == "chat":
                outputs = process_chat_output(input_ids=input_ids, output_ids_list=output_ids_list, 
                                            tokenizer=tokenizer, conv=conv)
            else:
                raise NotImplementedError
            
        except RuntimeError as e:
            print("ERROR question ID: ", task["task_idx"])
            print(e)
            outputs = ["ERROR"] * batch_size

        tasks_outputs.append({"task_idx": task["task_idx"], 
                              "task_id": task["task_id"], 
                              "outputs": outputs,
                              "gen_mode": gen_mode,
                              "temperature": temperature,
                              "max_new_tokens": max_new_tokens,
                              "tstamp": time.time(),
                              "model_id": model_id})

        # Dump answers
        with open(os.path.expanduser(save_generations_path), "a") as fout:
            for outputs in tasks_outputs:
                fout.write(json.dumps(outputs) + "\n")

def process_chat_output(input_ids, output_ids_list, conv, tokenizer):
    outputs = []
    for output_ids in output_ids_list:
        output_ids = output_ids[len(input_ids[0]) :]
        
        # be consistent with the template's stop_token_ids
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        if conv.name == "xgen" and output.startswith("Assistant:"):
            output = output.replace("Assistant:", "", 1).strip()
        outputs.append(output)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
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
    parser.add_argument("--save-generations-path", type=str, help="The output answer file.")
    parser.add_argument("--temperature", type=float, help="temperature.", default=0.2)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
        help="The maximum number of input tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="How many completion choices to generate in one batch.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default="bfloat16",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    run_generate(
        model_path=args.model_path,
        model_id=args.model_id,
        bench_name=args.bench_name,
        gen_mode=args.gen_mode,
        question_begin=args.question_begin,
        question_end=args.question_end,
        save_generations_path=args.save_generations_path,
        temperature=args.temperature,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
    )

    if args.save_generations_path:
        reorg_output_file(args.save_generations_path, args.num_samples)
