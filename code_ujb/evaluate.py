import argparse
import json
import time

from code_ujb import tasks

def run_evaluate(
    model_path,
    model_id,
    bench_name,
    gen_mode,
    question_begin,
    question_end,
    generations_file,
    output_file,
    num_samples,
):
    task_bench = tasks.get_task(bench_name)
    generations = json.load(open(generations_file, "r"))
    
    if question_begin>=0:
        generations = [g for g in generations if g["task_idx"] >= question_begin]
    if question_end>=0:
        generations = [g for g in generations if g["task_idx"] <= question_end]
    
    if num_samples:
        for idx in range(len(generations)):
            generations[idx]["outputs"] = generations[idx]["outputs"][:num_samples]
            
    if gen_mode == "complete":
        for generation in generations:
            outputs = task_bench.postprocess_complete_generations(generation["outputs"], generation["task_idx"])
            inputs = [task_bench.get_prompt_byidx(generation["task_idx"], mode="complete")]*len(outputs)
            raw_outputs = []
            for _input, pure_output, output in zip(inputs, outputs, generation["outputs"]):
                output.replace(_input, "")
                try:
                    index = output.index(pure_output)
                    raw_outputs.append(output[:index]+pure_output)
                except ValueError:
                    raw_outputs.append(output)
            generation["inputs"] = inputs
            generation["outputs"] = outputs
            generation["raw_outputs"] = raw_outputs
            
    elif gen_mode == "chat":
        for generation in generations:
            outputs = task_bench.postprocess_chat_generations(generation["outputs"], generation["task_idx"])
            inputs = [task_bench.get_prompt_byidx(generation["task_idx"], mode="chat")]*len(generation["outputs"])
            raw_outputs = []
            for _input, pure_output, output in zip(inputs, outputs, generation["outputs"]):
                output.replace(_input, "")
                try:
                    index = output.index(pure_output)
                    raw_outputs.append(output[:index]+pure_output)
                except ValueError:
                    raw_outputs.append(output)
            generation["inputs"] = inputs
            generation["outputs"] = outputs
            generation["raw_outputs"] = raw_outputs
    else:
        raise NotImplementedError()
    
    json.dump(generations, open(generations_file.replace(".json", "-postprocess.json"), "w"), indent=4)
            
    start_time = time.time()
    result = task_bench.evaluate(generations)
    end_time = time.time()
    result["time_cost"] = end_time - start_time
    result["end_time"] = end_time
    print(json.dumps(result, indent=4))
    json.dump(result, open(output_file, "w"), indent=4)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        required=True,
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
        default=-1,
    )
    parser.add_argument(
        "--question-end", 
        type=int, 
        help="A debug option. The end index of questions.",
        default=-1,
    )
    parser.add_argument("--load-generations-path", type=str, required=True, help="The output answer file.")
    parser.add_argument("--eval-output-path", type=str, help="The output answer file.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )

    args = parser.parse_args()
    
    run_evaluate(
        model_path=args.model_path,
        model_id=args.model_id,
        bench_name=args.bench_name,
        gen_mode=args.gen_mode,
        question_begin=args.question_begin,
        question_end=args.question_end,
        generations_file=args.load_generations_path,
        output_file=args.eval_output_path,
        num_samples=args.num_samples,
    )
