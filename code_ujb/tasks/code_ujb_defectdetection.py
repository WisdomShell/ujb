import os
import re

from code_ujb.Task import Task
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class StreamStopUJBDefectDetection():
    def __init__(self, mode="complete"):
        self.mode = mode
    
    def check_stop(self, generation):
        stop_str = ["Yes, it has defects", "No, it doesn't have defects"]
        for s in stop_str:
            if s in generation:
                return True
        return False

class CodeUJBDefectDetection(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "ZHENGRAN/code_ujb_defectdetection"

    def __init__(self):
        super().__init__(
            stop_words=[],
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
        return StreamStopUJBDefectDetection(mode=mode)
    
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["fix"]

    @staticmethod
    def _stop_at_function(generation):
        defect_stop_str = ["Yes, it has defect", "Yes", "has defect", "has defect"]
        no_defect_stop_str = ["No, it doesn't have defects", "No", 
                              "doesn't have defect", "do not have defect",
                              "do not has defect", "doesn't has defect"]
        for s in defect_stop_str:
            if s in generation:
                result = generation.split(s)[0]+s
                return result
        
        for s in no_defect_stop_str:
            if s in generation:
                result = generation.split(s)[0]+s
                return result
            
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
        generation = self._stop_at_function(generation)
        return generation

    def evaluate(self, generations):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        def check_defect(generation):
            defect_stop_str = [
                "Yes, it has defect", "Yes", "has defect", "has defect", "there are potential defects",
                "is potentially flawed",
                "There are defects",
                "There are several issue",
                "there is a potential defect"
            ]
            # 对缺陷的正则表达式
            defect_stop_regex = [
                r"here.*?potential issue",
                r"here.*?potential defect",
                r"There.*?a few issue",
                r"there.*?a few issue",
            ]

            no_defect_stop_str = [
                "no defects", "No, it doesn't have defects", "No", "doesn't have defect",
                "do not have defect", "do not has defect", "this function is not flawed"
            ]
            # 对没有缺陷的正则表达式
            no_defect_stop_regex = [
                "doesn't has defect",
                r"here are no.*?defect",
                r"do not see.*?defect",
                r"don't see.*?defect"
            ]

            for s in defect_stop_str:
                if s in generation:
                    result = generation.split(s)[0] + s
                    return "defect", result

            # 缺陷的正则表达式匹配
            for r in defect_stop_regex:
                regex_result = re.search(r, generation)
                if regex_result:
                    result = generation.split(regex_result.group())[0] + regex_result.group()
                    return "defect", result

            for s in no_defect_stop_str:
                if s in generation:
                    result = generation.split(s)[0] + s
                    return "nodefect", result

            # 没有缺陷的正则表达式匹配
            for r in no_defect_stop_regex:
                regex_result = re.search(r, generation)
                if regex_result:
                    result = generation.split(regex_result.group())[0] + regex_result.group()
                    return "nodefect", result
            
            no_result_stop_str = [
                "I cannot determine whether",
                "Before I provide my assessment",
                "I cannot determine",
                "determine if there are any defects",
                "determine whether there are any defects",
                "difficult to determine"
                "ERROR"
            ]
            for s in no_result_stop_str:
                if s in generation:
                    return "error", generation
                
            # if generation.startswith("The LM algorithm is a quasi-Newton"):
            #     return "error", generation
            # print("============================")
            # print("error", generation)
            # exit()
            return "error", generation
        def get_results(generations, num=-1):
            results = {"total": 0, "tp": 0, "tn": 0, 
                    "correct": 0, "fp": 0, "fn": 0, "error": 0}
            for generation in tqdm(generations, total=len(generations)):
                idx = generation["task_idx"]
                if num > 0:
                    gens = generation["outputs"][:num]
                else:
                    gens = generation["outputs"]
                
                for gen in gens:
                    label, gen = check_defect(gen)
                    if label == "error":
                        results["error"] += 1
                    elif label == "defect" and self.get_dataset()[idx]["defective"]:
                        results["correct"] += 1
                        results["tp"] += 1
                    elif label == "nodefect" and (not self.get_dataset()[idx]["defective"]):
                        results["correct"] += 1
                        results["tn"] += 1
                    elif label == "defect" and (not self.get_dataset()[idx]["defective"]):
                        results["fp"] += 1
                    elif label == "nodefect" and self.get_dataset()[idx]["defective"]:
                        results["fn"] += 1
                    
                    results["total"] += 1
            
            results["acc"] = results["correct"] / (results["total"]-results["error"])
            results["acc_w_error"] = results["correct"] / results["total"]
            results["precision"] = results["tp"] / (results["tp"]+results["fp"])
            results["recall"] = results["tp"] / (results["tp"]+results["fn"])
            results["f1"] = 2*(results["precision"]*results["recall"])/(results["precision"]+results["recall"])
            return results
        
        results_all = get_results(generations=generations, num=-1)
        results_one = get_results(generations=generations, num=1)
        
        return {"results_all": results_all, "results_one": results_one}
    
