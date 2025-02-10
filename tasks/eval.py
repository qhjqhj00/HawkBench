import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets
import json
import torch
import time
from tqdm import tqdm
from typing import Optional, Dict, List
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
import tiktoken
import numpy as np

from torch.utils.data import DataLoader

from tasks.prompts import *
from src import ModelArgs, DefaultDataCollator, FileLogger, makedirs, get_pipeline
from tasks.utils import scorer, set_seed, get_gpt_score, get_multichoice_score, get_sentence_level_score
from my_own_tools import *
from transformers.utils import logging

logger = logging.get_logger(__name__)
set_seed(1234)
level_prompts = {
    "level_1": QUESTION_ANSWERING_PROMPT_L1,
    "level_2": QUESTION_ANSWERING_PROMPT_L2,
    "level_3": QUESTION_ANSWERING_PROMPT_L3,
    "level_4": QUESTION_ANSWERING_PROMPT_L4
}
@dataclass
class Args(ModelArgs):
    eval_data_path: str = field(
        default="data/HawkBench",
        metadata={'help': 'The evaluation data path.'}
    )
    eval_level: List[str] = field(
        default_factory=lambda: ['1', '2', '3', '4'],
        metadata={'help': 'The evaluation level.'}
    )

    eval_datasets: List[str] = field(
        default_factory=lambda: [])

    output_dir: str = field(
        default="data/results/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Max input length.'}
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={'help': 'Truncate inputs from the middle.'}
    )



def process_hawkbench(data, indices, tokenizer, max_length=3500, truncate_from_middle=True):
    outputs = {'context': [], 'question': [], "dataset": [], "index": [], "length": [], "context_id": [], "identifier": []}

    for input, dataset, context, context_id, index, query_type in zip(data['input'], data['label'], data['context'], data["context_id"], indices, data["type"]):
        question = input
        if max_length is not None:
            if truncate_from_middle:
                try:
                    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
                except:
                    tokenized_context = tokenizer.encode(context)
                if len(tokenized_context) > max_length:
                    half = int(max_length / 2)
                    context = tokenizer.decode(tokenized_context[:half]) + tokenizer.decode(tokenized_context[-half:])
            else:
                tokenized_context = tokenizer.encode(context)
                context = tokenizer.decode(tokenized_context[-max_length:])

        length = len(tokenizer.encode(context))

        outputs["context"].append(context)
        outputs["context_id"].append(context_id)
        outputs["question"].append(question)
        outputs["dataset"].append(dataset)
        outputs["index"].append(index)
        outputs["length"].append(length)
        outputs["identifier"].append(f"{dataset}_{query_type}")
    return outputs

@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]
    accelerator = Accelerator(cpu=args.cpu)
    
    pipe = get_pipeline(args, device=accelerator.device)
    
    try:
        tokenizer = pipe.generator.tokenizer
    except:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    all_context = load_json(os.path.join(args.eval_data_path, "all_context.json"))
    with accelerator.main_process_first():
        process_fn = partial(
            process_hawkbench, 
            tokenizer=tokenizer,
            max_length=args.max_length,
            truncate_from_middle=args.truncate_from_middle
        )
        all_data_files = []
        for level in args.eval_level:
            eval_files = os.listdir(os.path.join(args.eval_data_path, f"level_{level}"))
            if args.eval_datasets:
                eval_files = [f for f in eval_files if f.split(".")[0] in args.eval_datasets]
            data_files = [os.path.join(f"{args.eval_data_path}/level_{level}", f) for f in eval_files]

            all_data_files.extend(data_files)
        raw_dataset = datasets.load_dataset("json", data_files=all_data_files, split="train")
        def add_context(example):
            example["context"] = all_context[example["context_id"]]["context"]
            return example
            
        raw_dataset = raw_dataset.map(add_context)
        dataset = raw_dataset.map(process_fn, batched=True, batch_size=50, num_proc=80, with_indices=True, remove_columns=raw_dataset.column_names)

    
    groupby_dataset = dataset.to_pandas().groupby("identifier")
    metrics = {}
    dataset_names = [key for key, _ in groupby_dataset]

    result_dir = os.path.join(args.output_dir, args.result_dir)

    for i, dataset_name in enumerate(dataset_names):
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {dataset_name} ({i + 1} / {len(dataset_names)})...")

        result_path = os.path.join(result_dir, f"{dataset_name}.json")
        

        dataset = datasets.Dataset.from_pandas(groupby_dataset.get_group(dataset_name), preserve_index=False)


        data_collator = DefaultDataCollator(padding_side=args.padding_side)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            collate_fn=data_collator,
            # only pin memory when no gpu
            pin_memory=not args.cpu,
        )

        # NOTE: prepare dataloader so the data moves to GPU automatically
        dataloader = accelerator.prepare(dataloader)

        indices = []
        preds = []    
        latencies = []        
        data_type = dataset_name.split("_")[1]
        for i, x in enumerate(tqdm(dataloader, desc="Generating")):

            # _prompt = level_prompts[f"level_{data_type}"] 
            _prompt = QUESTION_ANSWERING_PROMPT
            index = x.pop("index")[0]

            # NOTE: output should be a list
            start = time.time()            
            output = [pipe(x["context"][0], x["question"][0], prompt=_prompt, cache_id=x["context_id"][0], conv=args.conv)]
            latency = [time.time() - start]
            if accelerator.num_processes > 1:
                # pad across device to the same length
                output = accelerator.gather_for_metrics(output)
                index = accelerator.gather_for_metrics(index)
                latency = accelerator.gather_for_metrics(latency)

            accelerator.print(output)
            accelerator.print(f"\n{round(sum(latency) / len(latency), 2)} seconds per query.")

            index = index.tolist()

            if accelerator.process_index == 0:
                preds.extend(output)
                latencies.extend(latency)
                if isinstance(index, list):
                    indices.extend(index)
                else:
                    # single process
                    indices.append(index)

        if accelerator.process_index == 0:
            raw_dataset_subset = raw_dataset[indices]
            answers = raw_dataset_subset["answers"]
            questions = raw_dataset_subset["input"]
            selections = raw_dataset_subset["selections"]
            multichoice_answers = raw_dataset_subset["selection_answers"]
            
            score = scorer(preds, answers)        

            api_dict = load_json(args.api_dict)
            agent = Agent(args.agent_model, args.agent_provider, api_dict)
            level = dataset_name.split('_')[1]
            if level not in ["3", "4"]:
                multichoice_score, sent_precision, sent_recall = 0, 0, 0
            else:
                # multichoice_score = get_multichoice_score(agent, preds, selections, multichoice_answers)
                sent_precision, sent_recall = get_sentence_level_score(agent, answers, preds)

            # gpt_score = get_gpt_score(agent, questions, answers, preds)
            
            
            # score["gpt_score"] = gpt_score
            # score["multichoice_score"] = multichoice_score
            score["sent_precision"] = sent_precision
            score["sent_recall"] = sent_recall
            score["sent_f1"] = round((sent_precision+sent_recall)/2, 2)

            logger.info(f"{dataset_name}: {score}")
            average_latency = round(sum(latencies) / len(latencies),2)
            logger.info(f"latency: {average_latency} seconds per query")
            score["latency"] = average_latency
            metrics[dataset_name] = score

            with open(makedirs(result_path), "w", encoding="utf-8") as f:
                f.write(json.dumps(score, ensure_ascii=False) + "\n")
                for index, pred in zip(indices, preds):
                    sample = raw_dataset[index]
                    del sample["context"]
                    sample["pred"] = pred
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        # save config
        args.save(os.path.join(result_dir, "config.json"))
        
        level_metrics = defaultdict(list)
        for dataset_name, scores in metrics.items():
            # Extract level from dataset name (e.g. narrativeqa_level_1 -> level_1)
            level = dataset_name.split('_')[1]
            level_metrics[f'level_{level}'].append(scores)

        # Compute average for each level
        level_averages = {}
        for level, scores_list in level_metrics.items():
            level_avg = defaultdict(float)
            num_datasets = len(scores_list)
            
            # Sum up scores for each metric
            for scores in scores_list:
                for metric, value in scores.items():
                    level_avg[metric] += value
            
            # Calculate average
            for metric in level_avg:
                level_avg[metric] = round(level_avg[metric] / num_datasets, 2)
                
            level_averages[level] = dict(level_avg)
            metrics[f"avg_{level}"] = dict(level_avg)

        # Compute overall average across all datasets
        overall_avg = defaultdict(float)
        total_datasets = sum(len(scores_list) for scores_list in level_metrics.values())
        
        for scores_list in level_metrics.values():
            for scores in scores_list:
                for metric, value in scores.items():
                    overall_avg[metric] += value
                    
        for metric in overall_avg:
            overall_avg[metric] = round(overall_avg[metric] / total_datasets, 2)
            
        metrics["avg_overall"] = dict(overall_avg)

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args))
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            save_args = asdict(args)
            save_args["metrics"] = metrics
            f.write(json.dumps(save_args)+"\n")

if __name__ == "__main__":
    main()
