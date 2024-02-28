from pprint import pprint

from . import (
    code_ujb_repair, 
    code_ujb_complete, 
    code_ujb_testgen, 
    code_ujb_testgenissue,
    code_ujb_defectdetection)

TASK_REGISTRY = {
    "codeujbrepair": code_ujb_repair.CodeUJBRepair,
    "codeujbcomplete": code_ujb_complete.CodeUJBComplete,
    "codeujbtestgen": code_ujb_testgen.CodeUJBTestGen,
    "codeujbtestgenissue": code_ujb_testgenissue.CodeUJBTestGenIssue,
    "codeujbdefectdetection": code_ujb_defectdetection.CodeUJBDefectDetection,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
