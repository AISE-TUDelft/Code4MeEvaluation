import os
import json
from shutil import rmtree
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str, help="The directory to write the test sets to.",
                                default="output-unseen")
    parser.add_argument("models", type=str, help="Model to use. Supports multiple models",
                                nargs="+", choices=["unixcoder", "incoder", "codegpt_csn"])
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")
    MODELS = args.models

    models = []
    if "unixcoder" in MODELS:
        from unixcoder_predict import unixcoder
        models.append(unixcoder)
    if "incoder" in MODELS:
        from incoder_predict import incoder
        models.append(incoder)
    if "codegpt_java" in MODELS:
        from codegpt_java_predict import codegpt_java
        models.append(codegpt_java)
    if "codegpt_py" in MODELS:
        from codegpt_py_predict import codegpt_py
        models.append(codegpt_py)
    if "codegpt_csn" in MODELS:
        from codegpt_csn_predict import codegpt_csn
        models.append(codegpt_csn)

    if not os.path.exists(OUTPUT_DIR):
        raise Exception(f"Output directory {OUTPUT_DIR} does not exist. This directory should contain the test set .jsonl files.")

    for model in models:
        model_prediction_dir = os.path.join(PREDICTION_DIR, model["name"])
        if os.path.exists(model_prediction_dir):
            rmtree(model_prediction_dir)
        os.makedirs(model_prediction_dir)

    jsonl_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jsonl")]

    for jsonl_file in jsonl_files:
        print(f"Running models on {jsonl_file}...")
        with open(os.path.join(OUTPUT_DIR, jsonl_file), "r") as f:
            for sample in f:
                sample_obj = json.loads(sample)
                left_context = sample_obj["left_context"]
                gt = sample_obj["gt"]
                right_context = sample_obj["right_context"]

                for model in models:
                    output_file_name = f"{os.path.basename(jsonl_file).replace('.jsonl', '')}.txt"
                    output_file = os.path.join(PREDICTION_DIR, model["name"], output_file_name)
                    generate = model["generate"]
                    args = {}
                    if model["supports_left_context"]:
                        args["left_context"] = left_context
                    if model["supports_right_context"]:
                        args["right_context"] = right_context
                    prediction = generate(**args)
                    prediction_lines = prediction.splitlines()
                    prediction = "" if len(prediction_lines) == 0 else prediction_lines[0]
                    with open(output_file, "a") as f_out:
                        f_out.write(f"{prediction}\n")

if __name__ == "__main__":
    main()
