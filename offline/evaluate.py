import re
from argparse import ArgumentParser
import os
import json
from collections import defaultdict
from typing import List

from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

colors = ['#248af0', '#f5413b', '#ffb92f']

def main():
    plt.rcParams['figure.figsize'] = (6.4, 3.8)  # default is (6.4, 4.8)

    metric_names = list(compute_metrics("abc", "abc").keys())

    parser = ArgumentParser(description='Evaluator')
    parser.add_argument('-o', '--output_dir', default="output-unseen", type=str, help='Directory containing the test set .jsonl files and a predictions/ folder')
    parser.add_argument('-m', '--metrics', default=metric_names, nargs='+', choices=metric_names, help='Metrics to compute')
    parser.add_argument('-l', '--latex', action='store_true', help='Print LaTeX table')
    parser.add_argument('-p', '--plots-dir', default="plots", type=str, help='Directory to store plots')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} does not exist. This directory should contain the test set .jsonl files and a predictions/ folder")
        return

    if not os.path.exists(os.path.join(args.output_dir, "predictions")):
        print(f"Output directory {args.output_dir} does not contain a predictions/ folder. This directory should contain the test set .jsonl files and a predictions/ folder")
        return

    test_files = [f for f in os.listdir(args.output_dir) if f.endswith(".jsonl")]
    models = [f for f in os.listdir(os.path.join(args.output_dir, "predictions")) if os.path.isdir(os.path.join(args.output_dir, "predictions", f))]

    results = defaultdict(lambda: defaultdict(lambda: []))  # test file -> model -> dict[]
    results_by_trigger = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))  # test file -> model -> trigger -> dict[]
    results_by_language = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))  # test file -> model -> language -> dict[]

    for test_file in test_files:
        print(f"Evaluating {test_file}...")

        for model_idx, model in enumerate(models):
            prediction_file = os.path.join(args.output_dir, "predictions", model, test_file.replace(".jsonl", ".txt"))
            with open(prediction_file, "r") as f_pred, \
                 open(os.path.join(args.output_dir, test_file), "r") as f_test:
                for test_obj, prediction in zip(f_test, f_pred):
                    if prediction.strip() == "":
                        continue
                    test_obj = json.loads(test_obj)
                    prediction = prediction.strip()
                    gt = test_obj["gt"]
                    trigger = test_obj["trigger"]
                    language = test_obj["language"]

                    metrics = compute_metrics(gt, prediction)
                    results[test_file][model].append(metrics)
                    if trigger:
                        results_by_trigger[test_file][model][trigger.strip()].append(metrics)
                    results_by_language[test_file][model][language].append(metrics)

    print("Analysis done")

    print("Creating plots")
    os.makedirs(args.plots_dir, exist_ok=True)
    create_model_language_plots(results_by_language, args)
    create_triggerpoint_plots(results_by_trigger, args)

    if args.latex:
        print("Creating latex tables")

        print_model_avg_metrics_tables(results, args)
        print()

        print_trigger_avg_metrics_tables(results_by_trigger, args)
        print()

        print_language_avg_metrics_tables(results_by_language, args)
        print()


def create_model_language_plots(results_by_language, args):
    languages = [
        "Python",
        "Java",
        "TypeScript",
        "PHP",
        "JavaScript",
        "Ruby",
        "C++",
        "Rust",
        "C#",
        "Go",
        "C",
        "Scala",
    ]

    for test_file in results_by_language.keys():
        for plot_metric in args.metrics:
            models = list(results_by_language[test_file].keys())
            max_y = 0
            x = [i for i in range(len(languages))]
            width = 1 / (len(models) + 1)
            plt.ylabel(fix_metric(plot_metric))
            plt.xticks([i + width for i in x], languages, rotation=45)
            plt.yticks([i for i in range(0, 101, 10)])
            plt.yticks([i + 5 for i in range(0, 101, 10)], minor=True)
            for model_idx, model in enumerate(models):
                for language_idx, language in enumerate(languages):
                    if language in results_by_language[test_file][model]:
                        metric_values = results_by_language[test_file][model][language]
                        avg_metric_val = sum([m[plot_metric] for m in metric_values]) / len(metric_values)
                    else:
                        avg_metric_val = 0
                    plt.bar(x[language_idx] + model_idx * width, avg_metric_val, width * 0.75, color=colors[model_idx], label=re.match(r"^[a-z]+", model, re.IGNORECASE).group(0), zorder=3)
                    max_y = max(max_y, avg_metric_val)

            plt.ylim(0, max_y + 10)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=len(models))
            plt.tight_layout()
            plt.grid(axis='y', which='major', zorder=0, alpha=0.5, linestyle='-')
            plt.grid(axis='y', which='minor', zorder=0, alpha=0.5, linestyle='--')
            plt.savefig(os.path.join(args.plots_dir, f"languages-{test_file}-{plot_metric}.svg"))
            plt.clf()

def create_triggerpoint_plots(results_by_trigger, args):
    triggers = [
        ".",
        "(",
        "=",
        ",",
        "/",
        "-",
        "[",
        "<",
        "{",
        "if",
        ">",
        "return",
        "*",
        "+",
    ]

    for test_file in results_by_trigger.keys():
        for plot_metric in args.metrics:
            models = list(results_by_trigger[test_file].keys())
            max_y = 0
            x = [i for i in range(len(triggers))]
            width = 1 / (len(models) + 1)
            plt.ylabel(fix_metric(plot_metric))
            plt.xticks([i + width for i in x], triggers)
            plt.yticks([i for i in range(0, 101, 10)])
            plt.yticks([i + 5 for i in range(0, 101, 10)], minor=True)

            for model_idx, model in enumerate(models):
                for trigger_idx, trigger in enumerate(triggers):
                    if trigger in results_by_trigger[test_file][model]:
                        metric_values = results_by_trigger[test_file][model][trigger]
                        avg_metric_val = sum([m[plot_metric] for m in metric_values]) / len(metric_values)
                    else:
                        avg_metric_val = 0
                    plt.bar(x[trigger_idx] + model_idx * width, avg_metric_val, width * 0.75, color=colors[model_idx], label=re.match(r"^[a-z]+", model, re.IGNORECASE).group(0), zorder=3)
                    max_y = max(max_y, avg_metric_val)

            plt.ylim(0, max_y + 10)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=len(models))
            plt.tight_layout()
            plt.grid(axis='y', which='major', zorder=0, alpha=0.5, linestyle='-')
            plt.grid(axis='y', which='minor', zorder=0, alpha=0.5, linestyle='--')
            plt.savefig(os.path.join(args.plots_dir, f"triggers-{test_file}-{plot_metric}.svg"))
            plt.clf()


def print_model_avg_metrics_tables(results, args):
    for test_file in results.keys():
        table = r"""
\begin{table}[tb]
\centering
\caption{Metrics for """ + test_file + r"""}
\label{tab:metrics-""" + test_file + r"""}
\begin{tabular}{l""" + "r" * len(args.metrics) + r"""}
\toprule
\textbf{Model} & """ + " & ".join([rf"\textbf{{{m.upper()}}}" for m in args.metrics]) + r""" \\
\midrule
"""
        for model, metric_values in results[test_file].items():
            table += f"{model} & " + " & ".join(
                [f"{sum([m[metric_name] for m in metric_values]) / len(metric_values):.2f}" for metric_name in args.metrics]) + r" \\" + "\n"

        table = table.strip()
        table += r"""
\bottomrule
\end{tabular}
\end{table}"""

        print(table)

def print_trigger_avg_metrics_tables(results_by_trigger, args):
    for test_file in results_by_trigger.keys():
        for model, metric_values_by_trigger in results_by_trigger[test_file].items():
            table = r"""
\begin{table}[tb]
\centering
\caption{Metrics per trigger point for """ + model + ", " + test_file + r"""}
\label{tab:metrics-trigger-""" + model + "-" + test_file + r"""}
\begin{tabular}{ll""" + "r" * len(args.metrics) + r"""}
\toprule
\textbf{Trigger} & \textbf{n} & """ + " & ".join([rf"\textbf{{{m.upper()}}}" for m in args.metrics]) + r""" \\
\midrule
"""

            for trigger, metric_values in sorted(metric_values_by_trigger.items(), key=lambda x: len(x[1]), reverse=True):
                table += rf"\texttt{{{escape_latex(trigger.strip())}}} & {len(metric_values)} & " + " & ".join([f"{sum([m[metric_name] for m in metric_values]) / len(metric_values):.2f}" for metric_name in args.metrics]) + r" \\" + "\n"

            table = table.strip()
            table += r"""
\bottomrule
\end{tabular}
\end{table}"""
            print(table)

def print_language_avg_metrics_tables(results_by_language, args):
    for test_file in results_by_language.keys():
        for model in results_by_language[test_file].keys():
            table = r"""
\begin{table}[tb]
\centering
\caption{Metrics per language for """ + model + ", " + test_file + r"""}
\label{tab:metrics-language-""" + model + "-" + test_file + r"""}
\begin{tabular}{ll""" + "r" * len(args.metrics) + r"""}
\toprule
\textbf{Language} & \textbf{n} & """ + " & ".join([rf"\textbf{{{m.upper()}}}" for m in args.metrics]) + r""" \\
\midrule
"""

            for language, metric_values in sorted(results_by_language[test_file][model].items(), key=lambda x: len(x[1]), reverse=True):
                table += rf"\texttt{{{language}}} & {len(metric_values)} & " + " & ".join([f"{sum([m[metric_name] for m in metric_values]) / len(metric_values):.2f}" for metric_name in args.metrics]) + r" \\" + "\n"

            table = table.strip()
            table += r"""
\bottomrule
\end{tabular}
\end{table}"""
            print(table)

def compute_metrics(gt: str, pred: str):
    gt = gt.strip()
    pred = pred.strip()
    pred_tokens = tokenize(pred)
    gt_tokens = tokenize(gt)

    return {
        "em": 100 * (pred == gt),
        "es": fuzz.ratio(pred, gt),
        "bl": 100 * sentence_bleu([gt_tokens], pred_tokens, smoothing_function=SmoothingFunction().method2),
        "mr": 100 * meteor_score([gt_tokens], pred_tokens),
        "rl": 100 * rouge_l(gt_tokens, pred_tokens)
    }

tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
def tokenize(txt: str):
    return tokenizer.tokenize(txt)

def lcs(X: List[str], Y: List[str]) -> int:
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                continue
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

def rouge_l(true_tokens: List[str], prediction_tokens: List[str]) -> float:
    if len(true_tokens) == 0 or len(prediction_tokens) == 0:
        return 0
    lcs_length = lcs(true_tokens, prediction_tokens)
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(true_tokens)
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0

def escape_latex(txt: str) -> str:
    return txt.replace("_", r"\_") \
        .replace("{", r"\{") \
        .replace("}", r"\}") \
        .replace("#", r"\#") \
        .replace("&", r"\&") \
        .replace("$", r"\$") \
        .replace("%", r"\%") \
        .replace("[" , r"{[}") \
        .replace("]", r"{]}") \
        .replace("^", r"\textasciicircum{}") \
        .replace("~", r"\textasciitilde{}")


metric_dict = {
    "em": "Exact Match",
    "es": "Edit Similarity",
    "bl": "BLEU",
    "mr": "METEOR",
    "rl": "ROUGE-L"
}
def fix_metric(metric: str):
    if metric in metric_dict:
        return metric_dict[metric]
    return metric

if __name__ == "__main__":
    main()
