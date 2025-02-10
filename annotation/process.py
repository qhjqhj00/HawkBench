from .ql_tools import *
from .prompts import *
import os
import json

save_dir = "data/HawkBench"
source_dir = "data/HawkBench_processed"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from collections import Counter
level_counter = Counter()
all_context = load_json("data/HawkBench/all_context.json")
in_use_context = {}
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".jsonl"):
            level = root.split("/")[-1]
            data = load_jsonl(os.path.join(root, file))
            label = file.split(".")[0]
            for item in data:

                level_counter[item["level"]] += 1
                if item["context_id"] not in all_context:
                    print(item["context_id"])
                    continue
                if item["context_id"] not in in_use_context:
                    in_use_context[item["context_id"]] = all_context[item["context_id"]]
                if not os.path.exists(os.path.join(save_dir, f"{level}")):
                    os.makedirs(os.path.join(save_dir, f"{level}"))
                
                if item["level"] == 4:
                    selections = json.dumps(item["selections"])
                    selection_answers = json.dumps(item["selection_answers"])
                else:
                    selections = ""
                    selection_answers = ""

                with open(os.path.join(save_dir, f"{level}", f"{label}.jsonl"), "a") as f:
                    f.write(json.dumps({"input": item["question"], "answers": [item["answer"]], "context_id": item["context_id"], "_id": item["sample_id"], "label": label,  "type": item["level"], "selections": selections, "selection_answers": selection_answers}) + "\n")
save_json(in_use_context, os.path.join(save_dir, f"all_context.json"))
print(f"Saved {len(in_use_context)} context")

