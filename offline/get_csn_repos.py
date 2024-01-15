"""
Loads CodeSearchNet's train set, and writes all the repositories to 'codesearchnet_repos.txt"
"""

from datasets import load_dataset
from tqdm import tqdm

csn = load_dataset("code_search_net", split="train")

OUTPUT_FILE = "codesearchnet_repos.txt"

known_repos = set()

with open(OUTPUT_FILE, "w") as f:
    for sample in tqdm(csn, desc="Writing repositories", total=len(csn)):
        repository_name = sample["repository_name"]
        if repository_name in known_repos:
            continue
        f.write(f"{repository_name}\n")
        known_repos.add(repository_name)

print(f"Done! Wrote {len(known_repos)} repositories to {OUTPUT_FILE}")