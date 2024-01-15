import json
import os.path
from argparse import ArgumentParser
from collections import defaultdict


def main():
    parser = ArgumentParser()
    parser.add_argument('-o', '--output_dir', default="output-unseen", type=str, help='Output directory where test files are located')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Directory {args.outpur_dir} does not exist")
        exit(1)

    test_files = [f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir, f)) and f.endswith(".jsonl")]
    if len(test_files) == 0:
        print("No test files found")
        exit(0)

    for test_file in test_files:
        print(f"--- {test_file} ---")
        language_freq = defaultdict(lambda: 0)
        with open(os.path.join(args.output_dir, test_file), "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except:
                    continue
                language = obj['language']
                language_freq[language] += 1
                language_freq['all'] += 1

        for lang, freq in sorted(language_freq.items(), key=lambda x: x[1], reverse=True):
            print(f"{lang}: {freq}")


if __name__ == "__main__":
    main()
