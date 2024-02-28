# modified from MT-Bench
import os
import random
import openai
from text_generation import Client
import time
import json

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

def chat_compeletion_tgi(gen_mode, task, conv, temperature, max_tokens, stop_str_list):
    output = API_ERROR_OUTPUT
    API_URLS = os.environ[f"TGI_API_URL_{task['model_id'].replace('-', '_')}"]
    if API_URLS is None:
        print("Please set TGI_API_URL.")
    
    API_URLS = API_URLS.split(",")
    for _ in range(API_MAX_RETRY):
        try:
            api_url = random.choice(API_URLS)
            client = Client(api_url, timeout=60)
            if gen_mode == "chat":
                prompt = conv.get_prompt()
            else:
                prompt = task["question"]
                
            # response = client.generate(prompt, 
            #                            max_new_tokens=max_tokens, 
            #                            temperature=temperature, 
            #                            stop_sequences=stop_str_list,
            #                            truncate=3072)
            # output = response.generated_text
            
            output = ""
            for response in client.generate_stream(prompt, 
                                                   max_new_tokens=max_tokens, 
                                                   temperature=temperature, 
                                                   stop_sequences=stop_str_list,
                                                   truncate=3072):
                if not response.token.special:
                    output += response.token.text
                
                if task["stream_stop"].check_stop(output):
                    break
            
            if gen_mode == "complete":
                output = prompt + output
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_compeletion_openai(model, conv, temperature, max_tokens):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def reorg_output_file(save_generations_path, num_samples):
    def merged_output(a, b):
        a["outputs"].extend(b["outputs"])
        return a
    """Sort by question id and de-duplication"""
    answers = {}
    with open(save_generations_path+".tmp", "r") as fin:
        for l in fin:
            output = json.loads(l)
            tid = output["task_id"]
            if tid not in answers:
                answers[tid] = output
            else:
                answers[tid] = merged_output(answers[tid], output)

    qids = sorted(list(answers.keys()))
    answers = [answers[qid] for qid in qids]
    for answer in answers:
        answer["outputs"] = answer["outputs"][:num_samples]
    
    json.dump(answers, open(save_generations_path, "w"), indent=4)