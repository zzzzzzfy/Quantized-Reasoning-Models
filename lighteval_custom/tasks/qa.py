"""Custom evaluation tasks for LightEval."""

import random
import numpy as np
import re 
# from enum import Enum
 
from lighteval.metrics.metrics import Metrics

from lighteval.metrics.utils.metric_utils import SampleLevelMetric, MetricCategory, MetricUseCase
from lighteval_custom.metrics.dynamic_metrics import loglikelihood_acc_metric

from lighteval.metrics.normalizations import LogProbCharNorm

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# 构造prompt函数，仿照lighteval/tasks/default_prompts.py进行改写
def arc_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )
    
def hellaswag_preprocess(
    text: str,
    wikihow_artifacts: list[str] = [" [title]"],
    truncate_dots: bool = False,
    strip_text: bool = False,
    dot_replacement: str = ". ",
):
    """Comes from LM Eval Harness"""
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    for wikihow_artifact in wikihow_artifacts:
        text = text.replace(wikihow_artifact, dot_replacement)
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    if truncate_dots:
        text = text.replace(r"\.+", r"\.")
    if strip_text:
        text = text.strip()
    return text


def hellaswag_prompt_fn(line, task_name: str = None):
    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=hellaswag_preprocess(line["activity_label"] + ": " + ctx),
        choices=[hellaswag_preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
        # "metric": "choices_loglikelihood",
    )
    
def lambada_prompt_fn(line, task_name: str = None):
    query, choice = line["text"].rsplit(" ", 1)
    return Doc(
        task_name=task_name,
        query=query,
        gold_index=0,
        choices=[f" {choice}"],
    )

def piqa_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['goal']}\nAnswer:",
        choices=[f" {line['sol1']}", f" {line['sol2']}"],
        gold_index=int(line["label"]),
        # "metric": "choices_loglikelihood",
    )

def winogrande_prompt_fn(line, task_name: str = None):
    # LL of query + choices
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1']} {end_of_target}", f"{line['option2']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,  # managing unk test index
        # "metric": "choices_loglikelihood",
    )

# 从lighteval/tasks/default_tasks.py复制后修改
# 注意这里的指标要和reasoning.py中一样，尤其是metric，不能用复制来的metrics参数，lighteval库版本更新后进行了修改
arc_challenge = LightevalTaskConfig(
    name="arc:challenge",
    suite=["custom"],  # 仿照reasoning.py，使用默认的custom参数，实际意思是按照suite进行分类
    prompt_function=arc_prompt_fn,
    hf_repo="./datasets/ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metric=[loglikelihood_acc_metric(normalization=LogProbCharNorm(ignore_first_space=True))],
    stop_sequence=["\n"],
    version=0,
)

arc_easy = LightevalTaskConfig(
    name="arc:easy",
    suite=["custom"],
    prompt_function=arc_prompt_fn,
    hf_repo="./datasets/ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metric=[loglikelihood_acc_metric(normalization=LogProbCharNorm(ignore_first_space=True))],
    stop_sequence=["\n"],
    version=0,
)

hellaswag = LightevalTaskConfig(
    name="hellaswag",
    suite=["custom"],
    prompt_function=hellaswag_prompt_fn,
    hf_repo="./datasets/hellaswag/data",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metric=[loglikelihood_acc_metric(normalization=LogProbCharNorm())],
    stop_sequence=["\n"],
    version=0,
)

lambada_openai = LightevalTaskConfig(
    name="lambada:openai",
    suite=["custom"],
    prompt_function=lambada_prompt_fn,
    hf_repo="./datasets/lambada_openai",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    # 原本用的是ppl，但论文里用的是和其他一样的，但其他的是多选，这个换用单选的指标
    # metric=[Metrics.target_perplexity],
    metric=[Metrics.acc_golds_likelihood],
    stop_sequence=["\n"],
    version=0,
)

piqa = LightevalTaskConfig(
    name="piqa",
    suite=["custom"],
    prompt_function=piqa_prompt_fn,
    hf_repo="./datasets/piqa/data",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metric=[loglikelihood_acc_metric(normalization=LogProbCharNorm(ignore_first_space=True))],
    stop_sequence=["\n"],
    version=0,
)

winogrande = LightevalTaskConfig(
    name="winogrande",
    suite=["custom"],
    prompt_function=winogrande_prompt_fn,
    hf_repo="./datasets/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=-1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(arc_challenge)
TASKS_TABLE.append(arc_easy)
TASKS_TABLE.append(hellaswag)
TASKS_TABLE.append(lambada_openai)
TASKS_TABLE.append(piqa)
TASKS_TABLE.append(winogrande)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
