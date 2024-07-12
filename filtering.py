import os
import json

import matplotlib.pyplot as plt
from topics import ALL_TOPICS
import argparse

"""
python filtering.py --input_dir /input/dir/path --file_name sensei-4c84.jsonl > log.txt
"""

# Seed dataset is a jsonl file
def load_instances(file_path: str):
    instances = []
    with open(file_path) as f:
        for line in f:
            instances.append(json.loads(line))   
    return instances

def parse_arguments():
    parser = argparse.ArgumentParser(description="Sensei Filtering")
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help='Input directory with sense files.'),
    parser.add_argument("--file_name",
                        type=str,
                        required=True,
                        help='Name of the file'),
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_arguments()
    input_dir = args.input_dir
    file_name = args.file_name

    input_file_path = os.path.join(input_dir, file_name)
    output_file_path = os.path.join(input_dir, f"clean-{file_name}")
    
    instances = load_instances(file_path=input_file_path)

    updates_count = 0
    SUBJECT_AREA_STR = ["Subject Area:","SUBJECT_AREA:", "SUBJECT AREA:", "subject area:"]
    QUESTION_STR = ["Question:","QUESTION:"]
    for instance in instances:
        filtered = ""
        for topic in ALL_TOPICS:
            if topic in instance["input"]:
                filtered = instance["input"].replace(topic,"").strip()

                # Clean 'Subject Area:' and variants (if preset)
                for str in SUBJECT_AREA_STR:
                    if filtered.startswith(str): 
                        filtered = filtered.replace(str,"",1).strip()
                        break
                
                # Clean 'Question:' and variants (if present)
                for str in QUESTION_STR:
                    if filtered.startswith(str):
                        filtered = filtered.replace(str,"",1).strip()
                        break
                # Clean leading and tailing '' (if present)
                if filtered.startswith("'") and filtered.endswith("'"):
                    filtered = filtered[1:-1].strip()
                                # Clean leading and tailing '' (if present)
                if filtered.startswith("\"") and filtered.endswith("\""):
                    filtered = filtered[1:-1].strip()

                # TEST
                print("= Before =")
                print(instance["input"])
                print("= After =")
                print(filtered)
                updates_count += 1
                break
    
        if len(filtered)>0:
            instance["input"] = filtered

        # Save instance to output file
        with open(output_file_path, "a") as output_file:
            output_file.write(json.dumps(instance) + "\n")
            
    # Save clean file
    print(f"Sample updates: {updates_count}")
    print(f"Total updates: {len(instances)}")
    print(f"Output file saved at: {output_file_path}")



