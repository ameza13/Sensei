import os
import json

# Seed dataset is a jsonl file
def load_instances(file_path: str):
    instances = []
    with open(file_path) as f:
        for line in f:
            instances.append(json.loads(line))   
    return instances

dataset_path = 'jsonl/file/path'
instances = load_instances(file_path=dataset_path)

min_len = 4000
max_len = 0
empty_response = 0
for instance in instances:
    if min_len > len(instance["output"]): min_len = len(instance["output"])
    if max_len < len(instance["output"]): max_len = len(instance["output"])
    if len(instance["output"]) == 0: empty_response+=1

print(f"Dataset: {dataset_path}")
print(f"# Instances: {len(instances)}")
print(f"min_len: {min_len}")
print(f"max_len: {max_len}")
print(f"Empty Responses: {empty_response}")