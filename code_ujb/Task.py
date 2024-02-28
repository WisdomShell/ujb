from abc import ABC, abstractmethod
from warnings import warn

from datasets import load_dataset

def remove_line_comment(signature):
    pure_signature = ""
    line_comment = False
    for idx, c in enumerate(signature):
        if c == "/" and idx<len(signature)-1 and signature[idx+1] == "/":
            line_comment = True
        if line_comment:
            if c == "\n":
                line_comment = False
            continue
        pure_signature += c
    return pure_signature

def clean_signature(signature):
    signature = remove_line_comment(signature)
    if signature.startswith("@"):
        for idx, c in enumerate(signature):
            if c == " " or c == "\n":
                break
        pre_signature = signature[:idx]+"\n"
        sub_signature = signature[idx:].strip()
        sub_signature = sub_signature.split("(")[0].strip()
    else:
        pre_signature = ""
        sub_signature = signature.split("(")[0]
    return pre_signature, sub_signature

class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            if hasattr(self, "dataset") and self.dataset is None: raise Exception("Use locally downloaded dataset.")
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        except:
            warn(
                "This task will use a locally downloaded dataset, not from the HF hub."
            )

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_complete_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass
    
    @abstractmethod
    def postprocess_chat_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def evaluate(self, generations):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :return: dict[str: float]
        """
        pass
