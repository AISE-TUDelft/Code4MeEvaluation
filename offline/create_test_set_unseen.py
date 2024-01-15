import os
import json
import random
import argparse
from typing import List
import re
from pathlib import Path

UNSEEN_PATH = "Code-sampled100"
TRIGGER_POINTS = [
    "await ",
    "assert ",
    "raise ",
    "del ",
    "lambda ",
    "yield ",
    "return ",
    "while ",
    "for ",
    "if ",
    "elif ",
    "else ",
    "global ",
    "in ",
    "and ",
    "not ",
    "or ",
    "is ",
    "with ",
    "except ",
    ".",
    "+",
    "-",
    "*",
    "/",
    "%",
    "**",
    "<<",
    ">>",
    "&",
    "|",
    "^",
    "==",
    "!=",
    "<=",
    ">=",
    "+=",
    "-=",
    "=",
    "<",
    ">",
    ";",
    ",",
    "[",
    "(",
    "{",
    "~"
]

# minimum distance between chosen code completions
MIN_CHAR_CLEARANCE = 50

def unseen_file_iterator():
    for language in os.listdir(UNSEEN_PATH):
        for user in os.listdir(os.path.join(UNSEEN_PATH, language)):
            for repo in os.listdir(os.path.join(UNSEEN_PATH, language, user)):
                for file in os.listdir(os.path.join(UNSEEN_PATH, language, user, repo)):
                    yield os.path.join(UNSEEN_PATH, language, user, repo, file)


MODE_RANDOM = "random"
MODE_TRIGGER_POINT = "triggerpoint"

def extract_indices(mode: str, content: str) -> List[object]:
    indices = []
    if mode == MODE_RANDOM:
        space_pat = " +"
        for match in re.finditer(space_pat, content):
            # left context on this same line must be non-empty
            last_newline_idx = max(0, content.rfind("\n", 0, match.start()))
            before = content[last_newline_idx:match.start()]
            if before.strip() == "":
                continue

            # right context on this same line must be non-empty (otherwise we have nothing to predict but whitespaces)
            next_newline_idx = content.find("\n", match.end())
            after = content[match.end():next_newline_idx]
            if after.strip() == "":
                continue

            indices.append({"index": match.end()})
    elif mode == MODE_TRIGGER_POINT:
        for trigger in TRIGGER_POINTS:
            start = 0
            while start < len(content):
                start = content.find(trigger, start)
                if start == -1:
                    break
                start += len(trigger)

                # right context on this same line must be non-empty (otherwise we have nothing to predict but whitespaces)
                next_newline_idx = content.find("\n", start)
                after = content[start:next_newline_idx]
                if after.strip() == "":
                    continue

                indices.append({"index": start, "trigger": trigger})
    return indices

def main():
    parser = argparse.ArgumentParser(description='Test set generator')
    parser.add_argument('-o', '--output_dir', default="output-unseen", type=str, help='Output directory')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('-m', '--mode', required=True, type=str, help='Code completion task mode', choices=[MODE_RANDOM, MODE_TRIGGER_POINT])
    parser.add_argument('-s', '--max_samples', default=10, type=int, help='Maximum number of samples per file')
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite of existing output directory')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    file_count = 0
    valid_file_count = 0
    sample_count = 0

    output_file = os.path.join(args.output_dir, f"output-{args.mode}.jsonl")
    if os.path.exists(output_file):
        if args.force:
            os.remove(output_file)
        else:
            print(f"Error: Output file '{output_file}' already exists. Use --force to overwrite.")
            return

    with open(output_file, 'w') as f_out:
        for file_path in unseen_file_iterator():
            file_count += 1
            with open(file_path, 'r') as f_in:
                content = f_in.read()
            indices = extract_indices(args.mode, content)
            if not indices:
                continue
            valid_file_count += 1

            samples = []

            while len(samples) < args.max_samples and indices:
                sample = random.choice(indices)
                samples.append(sample)
                indices = [obj for obj in indices if abs(obj["index"] - sample["index"]) > MIN_CHAR_CLEARANCE]

            sample_count += len(samples)

            for i, sample in enumerate(samples):
                idx = sample["index"]
                left_context = content[:idx].rstrip()
                eol_idx = content.find("\n", idx)
                gt = content[idx:eol_idx].strip()
                right_context = content[eol_idx:]
                json.dump({
                    "left_context": left_context,
                    "gt": gt,
                    "right_context": right_context,
                    "file_path": file_path,
                    "prediction_idx": i,
                    "trigger": sample["trigger"] if "trigger" in sample else None,
                    "language": Path(file_path).parent.parent.parent.name,
                }, f_out)
                f_out.write('\n')

    print(f"Total files: {file_count}")
    print(f"Valid files: {valid_file_count}")
    print(f"Total samples: {sample_count}")

if __name__ == "__main__":
    main()
