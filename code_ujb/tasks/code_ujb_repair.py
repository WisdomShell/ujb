import os
import random
import re
import signal
import string
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import chardet
import javalang
import numpy as np
from code_ujb.Task import Task, clean_signature
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class StreamStopUJBRepair():
    def __init__(self, function_signature, mode="complete"):
        self.function_signature = function_signature
        self.mode = mode
    
    def check_stop(self, generation):
        block_count, in_block, in_double_quote, in_single_quote = 0, False, False, False
        for char_idx in range(len(generation)):
            if generation[char_idx] == '"': in_double_quote = not in_double_quote
            if generation[char_idx] == "'": in_single_quote = not in_single_quote
            if generation[char_idx] == "{" and (not in_double_quote): 
                block_count += 1
                in_block = True
            if generation[char_idx] == "}" and (not in_double_quote): 
                block_count -= 1
            if block_count == 0 and in_block:
                return True
        return False

class CodeUJBRepair(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "ZHENGRAN/code_ujb_repair"

    def __init__(self):
        super().__init__(
            stop_words=["// Provide a fix for the buggy function", 
                        '// Buggy Function', '// Fixed Function', 
                        '# Buggy Function', '# Fixed Function',
                        '/* Buggy Function */', '/* Fixed Function */'],
            requires_execution=False,
        )
        print("Using Dataset:", self.DATASET_PATH)
        self.dataset = load_dataset(self.DATASET_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["train"]

    def get_prompt(self, doc, mode="complete"):
        """Builds the prompt for the LM to generate from."""
        if mode == "complete":
            prompt_key = "prompt_complete"
        elif mode == "chat":
            prompt_key = "prompt_chat"
        else:
            raise KeyError()
        return doc[prompt_key].strip()
    
    def get_prompt_byidx(self, idx, mode="complete"):
        """Builds the prompt for the LM to generate from."""
        return self.get_prompt(self.get_dataset()[idx], mode=mode)

    def get_id_byidx(self, idx):
        """Builds the prompt for the LM to generate from."""
        return self.get_dataset()[idx]["task_id"]
    
    def get_stream_stop(self, idx, mode="complete"):
        return StreamStopUJBRepair(self.get_dataset()[idx]["function_signature"], mode=mode)
    
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["fix"]

    @staticmethod
    def _stop_at_function(generation):
        block_count, in_block, in_double_quote, in_single_quote = 0, False, False, False
        char_idx = 0
        for char_idx in range(len(generation)):
            if generation[char_idx] == '"': in_double_quote = not in_double_quote
            if generation[char_idx] == "'": in_single_quote = not in_single_quote
            if generation[char_idx] == "{" and (not in_double_quote): 
                block_count += 1
                in_block = True
            if generation[char_idx] == "}" and (not in_double_quote): 
                block_count -= 1
            if block_count == 0 and in_block:
                break
        if char_idx:
            generation = generation[:char_idx+1]
        return generation

    def postprocess_complete_generations(self, generations, idx):
        return [self.postprocess_complete_generation(gen, idx) for gen in generations]
    
    def postprocess_chat_generations(self, generations, idx):
        return [self.postprocess_chat_generation(gen, idx) for gen in generations]
    
    def postprocess_complete_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        # prompt = self.get_prompt(self.dataset["train"][idx])
        prompt = self.dataset["train"][idx]["prompt_complete"]
        generation = generation[len(prompt):]
        generation = self._stop_at_function(generation)
        # print("function", self.dataset["train"][idx]["fix"])
        # print("generation", generation)
        return generation
    
    def postprocess_chat_generation(self, generation, idx):
        signature = self.dataset["train"][idx]["function_signature"].strip()
        
        pre_signature, sub_signature = clean_signature(signature)
        if not sub_signature in generation:
            print("Can not find target function in answer!")
            return "Can not find target function in answer!\n\n"+generation
        generation = generation.split(sub_signature)
        # if len(generation) != 2:
        #     print("Multiple target function in answer!")
        #     return "Multiple target function in answer!\n\n"+generation
        generation = generation[1]
        function = self._stop_at_function(generation)
        
        generation = pre_signature + sub_signature + function
        return generation

    def evaluate(self, generations):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        all_tasks = []
        results = {"total": 0, "pass_syntax": {"count": 0}, "pass_compile": {"count": 0}, 
                   "pass_trigger": {"count": 0}, "pass_all": {"count": 0}, "timed_out": 0, "detail": {}}
        total_tokens_dict = {}
        for generation in tqdm(generations, total=len(generations)):
            idx = generation["task_idx"]
            gens = generation["outputs"]
            
            inps = generation["inputs"]
            rawouts = generation["raw_outputs"]
            inps_tokens = sum([len(tokens) for tokens in self.tokenizer.batch_encode_plus(inps, return_tensors="np")['input_ids']])
            rawouts_tokens = sum([len(tokens) for tokens in self.tokenizer.batch_encode_plus(rawouts, return_tensors="np")['input_ids']])
            total_tokens_dict[idx] = inps_tokens * 0.5 + rawouts_tokens
            
            project = self.dataset["train"][idx]["project"]
            bug_id = self.dataset["train"][idx]["bug_id"]
            bug_key = f"{project}-{bug_id}"
            testmethods = self.dataset["train"][idx]["testmethods"]
            # testmethods = testmethods[:1]
            source_dir = self.dataset["train"][idx]["source_dir"]
            start = self.dataset["train"][idx]["start"]
            end = self.dataset["train"][idx]["end"]
            location = self.dataset["train"][idx]["location"]
            source = self.dataset["train"][idx]["source"]
            
            one_tasks = [(idx, gen, project, bug_id, testmethods, source_dir, 
                      start, end, location, source) for gen in gens]
            all_tasks.extend(one_tasks)
        
        # p = Pool(1) 
        with ProcessPoolExecutor(max_workers=os.cpu_count()//4) as executor:
            # Submit all your tasks to the executor
            future_tasks = set()
            for task in all_tasks:
                future_tasks.add(executor.submit(validate_all_patches, task))
                time.sleep(0.01)
            # Use tqdm to display progress
            all_bug_results_list = []
            with tqdm(as_completed(future_tasks), total=len(all_tasks), desc="Evaluating all tasks...") as progress_bar:
                for future in progress_bar:
                    # Append the result to a list
                    all_bug_results_list.append(future.result())
        
        all_bug_results_dict = {}
        for bug_result in all_bug_results_list:
            if bug_result["idx"] not in all_bug_results_dict:
                all_bug_results_dict[bug_result["idx"]] = []
            all_bug_results_dict[bug_result["idx"]].append(bug_result)
        
        keys_list = list(all_bug_results_dict.keys())
        keys_list.sort()
        for idx in keys_list:
            bug_results = all_bug_results_dict[idx]
            
            example_detail = {"total": 0, "total_tokens": total_tokens_dict[idx], 
                              "pass_syntax": {"count": 0}, "pass_compile": {"count": 0}, 
                              "pass_trigger": {"count": 0}, "pass_all": {"count": 0}, "timed_out": 0, }
            for detail in bug_results:
                example_detail["total"] += 1
                if detail["pass_syntax"]:
                    example_detail["pass_syntax"]["count"] += 1
                if detail["pass_compile"]:
                    example_detail["pass_compile"]["count"] += 1
                if detail["pass_trigger"]:
                    example_detail["pass_trigger"]["count"] += 1
                if detail["pass_all"]:
                    example_detail["pass_all"]["count"] += 1
                if detail["timed_out"]:
                    example_detail["timed_out"] += 1
            
            for key in list(results.keys()):
                if not "pass" in key: continue
                for k in [1, 5, 10, 20, 100]:
                    if example_detail["total"] < k: continue 
                    example_detail[key][f"pass@k-{k}"] = get_pass_at_k(example_detail["total"], example_detail[key]["count"], k)
                    if not f"pass@k-{k}" in results[key]:
                        results[key][f"pass@k-{k}"] = []
                    results[key][f"pass@k-{k}"].append(example_detail[key][f"pass@k-{k}"])
                for t in [1000, 5000, 10000, 20000, 500000, 1000000]:
                    if example_detail["total_tokens"] < t: continue 
                    tk = t / (example_detail["total_tokens"] / example_detail["total"])
                    example_detail[key][f"pass@t-{t}"] = get_pass_at_k(example_detail["total"], example_detail[key]["count"], tk)
                    if not f"pass@t-{t}" in results[key]:
                        results[key][f"pass@t-{t}"] = []
                    results[key][f"pass@t-{t}"].append(example_detail[key][f"pass@t-{t}"])
                    
            print(example_detail)
            results["detail"][idx] = example_detail
            results["total"] += 1
            if example_detail["pass_syntax"]["count"] > 0:
                results["pass_syntax"]["count"] += 1
            if example_detail["pass_compile"]["count"] > 0:
                results["pass_compile"]["count"] += 1
            if example_detail["pass_trigger"]["count"] > 0:
                results["pass_trigger"]["count"] += 1
            if example_detail["pass_all"]["count"] > 0:
                results["pass_all"]["count"] += 1
            results["timed_out"] += example_detail["timed_out"]
        
        for key in list(results.keys()):
            if not "pass" in key: continue
            for pkey in list(results[key].keys()):
                if "@" in pkey:
                    results[key][pkey] = np.mean(results[key][pkey])
        return results
    
def get_pass_at_k(n, c, k):
    if n - c < k : return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def read_file(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    encoding = chardet.detect(content)['encoding']
    decoded_content = content.decode(encoding)
    return decoded_content

def save_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def validate_all_patches(item):
    idx, patch, project, bug_id, testmethods, source_dir, start, end, location, source = item
    def generate_random_string(length):
        characters = string.ascii_letters + string.digits  # 包含大写字母、小写字母和数字
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string

    tmp_folder = f"{project}-{bug_id}-" + generate_random_string(8)
    subprocess.run(['rm', '-rf', '/tmp/' + tmp_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd = ['defects4j', 'checkout', '-p', project, '-v', str(bug_id) + 'f', '-w', '/tmp/' + tmp_folder]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    raw_source = source
    source = source.split("\n")
    patch = patch.split("\n")
    source = "\n".join(source[:start] + patch + source[end+1:])
    # print("Diff")
    # print(get_unified_diff(raw_source, source))

    save_file("/tmp/" + tmp_folder + "/" + location, source)
    
    compile_fail, timed_out, bugg, entire_bugg, syntax_error = run_d4j_test(source, testmethods, tmp_folder)
        
    subprocess.run(['rm', '-rf', '/tmp/' + tmp_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return {
            "idx": idx,
            "project": project,
            "bug_id":  bug_id,
            "pass_syntax": not syntax_error,
            "pass_compile": not compile_fail,
            "pass_trigger": not bugg,
            "pass_all": not entire_bugg,
            "timed_out": timed_out
        }
    
def run_d4j_test(source, testmethods, bug_id):
    bugg = False
    compile_fail = False
    timed_out = False
    entire_bugg = True
    error_string = ""


    try:
        tokens = javalang.tokenizer.tokenize(source)
        parser = javalang.parser.Parser(tokens)
        parser.parse()
    except:
        # print("Syntax Error")
        return True, False, True, True, True

    for t in testmethods:
        # print(t.strip())
        cmd = 'defects4j test -w %s/ -t %s' % (('/tmp/' + bug_id), t.strip())
        Returncode = ""
        error_file = open("/tmp/stderr.txt", "wb")
        # print(cmd)
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                # print(b"".join(Returncode).decode('utf-8'))
                error_file.close()
                break
            elif Flag != 0 and Flag is not None:
                compile_fail = True
                error_file.close()
                with open("/tmp/stderr.txt", "rb") as f:
                    r = f.readlines()
                for line in r:
                    if re.search(':\serror:\s', line.decode('utf-8')):
                        error_string = line.decode('utf-8')
                        break
                # print("error_string", error_string)
                break
            elif time.time() - while_begin > 180:
                error_file.close()
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                timed_out = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            continue
        else:
            bugg = True
            break

    # Then we check if it passes all the tests, include the previously okay tests
    if not bugg:
        # print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
        cmd = 'defects4j test -w %s/' % ('/tmp/' + bug_id)
        Returncode = ""
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
                                start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                break
            elif Flag != 0 and Flag is not None:
                bugg = True
                break
            elif time.time() - while_begin > 240:
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                bugg = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            entire_bugg = False

    return compile_fail, timed_out, bugg, entire_bugg, False
