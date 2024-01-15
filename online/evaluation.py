import json
from argparse import ArgumentParser
import os
from collections import defaultdict
from typing import List
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer
from matplotlib import pyplot as plt
from tqdm import tqdm

colors = ['#248af0', '#f5413b', '#ffb92f']

def main():
    plt.rcParams['figure.figsize'] = (6.4, 3.8)  # default is (6.4, 4.8)

    metric_names = list(compute_metrics("abc", "abc").keys())

    parser = ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to data folder')
    parser.add_argument('-p', '--plots-dir', type=str, default='plots', help='Path to plots folder')
    parser.add_argument('-m', '--metrics', default=metric_names, nargs='+', choices=metric_names, help='Metrics to compute')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f'Path {args.data} does not exist')
        exit(1)

    os.makedirs(args.plots_dir, exist_ok=True)

    languages = ["python", "java", "typescript", "php", "javascript", "kotlin", "cpp", "rust", "csharp", "go", "c", "scala"]

    total_files = 0  # total number of data files
    valid_files = 0  # number of data files that are valid
    accepted_prediction_count = 0
    acceptance_per_language = defaultdict(lambda: defaultdict(lambda: 0))
    acceptance_per_model_per_language = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    context_size_per_language = defaultdict(lambda: defaultdict(lambda: []))
    triggerpoint_freq = defaultdict(lambda: 0)  # over valid data
    model_freq = defaultdict(lambda: 0)  # over all data
    ide_lang_freq = defaultdict(lambda: defaultdict(lambda: 0))  # over all data
    model_lang_freq = defaultdict(lambda: defaultdict(lambda: 0))  # over all data
    lang_freq = defaultdict(lambda: defaultdict(lambda: 0))  # over all data
    empty_preds_total = 0
    empty_preds_per_model_per_language = defaultdict(lambda: defaultdict(lambda: 0))
    empty_gt_total = 0
    empty_gt_per_model_per_language = defaultdict(lambda: defaultdict(lambda: 0))
    shared_context_size_per_language = defaultdict(lambda: defaultdict(lambda: []))
    total_predictions = 0
    valid_predictions = 0

    results_by_trigger = defaultdict(lambda: defaultdict(lambda: []))  # model -> trigger -> dict[]
    results_by_language = defaultdict(lambda: defaultdict(lambda: []))  # model -> language -> dict[]
    accepted_results_by_trigger = defaultdict(lambda: defaultdict(lambda: []))  # model -> trigger -> dict[]
    accepted_results_by_language = defaultdict(lambda: defaultdict(lambda: []))  # model -> language -> dict[]
    mode_per_ide = defaultdict(lambda: defaultdict(lambda: 0))  # ide -> {trigger: 0, manual: 0}

    users_with_context = set()

    for file in tqdm(os.listdir(args.data)):
        if not file.endswith(".json"):
            continue
        user_id = file.split('-')[0]
        # if total_files > 10000:
        #     break
        total_files += 1
        with open(os.path.join(args.data, file), 'r') as f:
            try:
                data = json.load(f)
                translate_data(data)  # ensure compatibility with the latest format
            except:
                continue

        models = get_models(data)
        model_predictions = get_model_predictions(data)
        language = get_language(data['language'])
        trigger = None if 'keybind' in data and data['keybind'] else data['triggerPoint']

        if not 'groundTruth' in data or data['groundTruth'] is None or data['groundTruth'].strip() == '':
            empty_gt_total += 1
            for model in models:
                empty_gt_per_model_per_language[model][language] += 1
                empty_gt_per_model_per_language[model]['all'] += 1

        if 'modelPredictions' not in data or 'predictions' not in data or all(pred.strip() == '' for pred in data['predictions']):
            empty_preds_total += 1
            for model in models:
                empty_preds_per_model_per_language[model][language] += 1
                empty_preds_per_model_per_language[model]['all'] += 1

        lang_freq[language]["total"] += len(models)

        total_predictions += len(model_predictions)

        model_valid_predictions = {
            model: prediction
            for model, prediction in model_predictions.items()
            if prediction.strip() != ''
        }

        valid_predictions += len(model_valid_predictions)

        lang_freq[language]["valid"] += 1

        if trigger is not None:
            mode_per_ide[data['ide']]["trigger"] += 1
            mode_per_ide['all']["trigger"] += 1
        else:
            mode_per_ide[data['ide']]["manual"] += 1
            mode_per_ide['all']["manual"] += 1

        ide_lang_freq[data['ide']][language] += len(models)
        ide_lang_freq[data['ide']]['all'] += len(models)

        if 'leftContextLength' in data and 'rightContextLength' in data:
            left_context_size = data['leftContextLength']
            right_context_size = data['rightContextLength']
            context_size_per_language[language]["left"].append(left_context_size)
            context_size_per_language[language]["right"].append(right_context_size)
            context_size_per_language['all']["left"].append(left_context_size)
            context_size_per_language['all']["right"].append(right_context_size)

        if 'leftContext' in data and data['leftContext'] is not None and 'rightContext' in data and data['rightContext'] is not None:
            users_with_context.add(user_id)
            left_context_tokens = len(data['leftContext'].split())
            right_context_tokens = len(data['rightContext'].split())
            shared_context_size_per_language[language]["left"].append(left_context_tokens)
            shared_context_size_per_language[language]["right"].append(right_context_tokens)
            shared_context_size_per_language['all']["left"].append(left_context_tokens)
            shared_context_size_per_language['all']["right"].append(right_context_tokens)

        for model in models:
            model_freq[model] += 1
            model_lang_freq[model][language] += 1
            model_lang_freq[model]['all'] += 1

        if not is_valid(data):
            continue

        valid_files += 1
        gt = data['groundTruth']
        accepted_prediction = get_accepted_prediction(data)

        acceptance_per_language[language]["total"] += 1
        acceptance_per_language["all"]["total"] += 1

        if accepted_prediction is not None:
            acceptance_per_language[language]["accepted"] += 1
            acceptance_per_language["all"]["accepted"] += 1

        for model in models:
            if model not in model_valid_predictions:
                continue

            accepted = accepted_prediction is not None and accepted_prediction == model_valid_predictions[model]

            acceptance_per_model_per_language[model][language]["total"] += 1
            acceptance_per_model_per_language[model]["all"]["total"] += 1
            if accepted:
                acceptance_per_model_per_language[model][language]["accepted"] += 1
                acceptance_per_model_per_language[model]["all"]["accepted"] += 1

            model_metric = compute_metrics(gt, model_valid_predictions[model])

            if trigger is not None:
                results_by_trigger[model][trigger].append(model_metric)

                triggerpoint_freq['all'] += 1
                triggerpoint_freq[trigger] += 1

                if accepted:
                    accepted_results_by_trigger[model][trigger].append(model_metric)

            results_by_language[model][language].append(model_metric)

            if accepted:
                accepted_results_by_language[model][language].append(model_metric)

        if accepted_prediction is not None:
            accepted_prediction_count += 1

    print(f"Total number of files: {total_files}")
    print(f"Number of valid files: {valid_files}")
    print(f"Total number of predictions (all models): {total_predictions}")
    print(f"Number of valid predictions (all models): {valid_predictions}")
    print()

    print(f"Users that have shared context: {len(users_with_context)}")
    print()

    print("Trigger points vs manual completion per IDE:", mode_per_ide)
    for ide in mode_per_ide:
        total = mode_per_ide[ide]['trigger'] + mode_per_ide[ide]['manual']
        print(f"{ide}: {mode_per_ide[ide]['trigger']}/{mode_per_ide[ide]['manual']}, {100 * mode_per_ide[ide]['trigger'] / total:.2f}% is trigger point, {100 * mode_per_ide[ide]['manual'] / total:.2f}% is manual")
    print()

    print("Mean context size per language:")
    for language in ["all", *set(languages).intersection(context_size_per_language.keys())]:
        print(f"{language}: {mean(context_size_per_language[language]['left']):.2f} left, {mean(context_size_per_language[language]['right']):.2f} right")
    print()

    print("Mean shared context size per language (in space-split tokens):")
    for language in ["all", *set(languages).intersection(shared_context_size_per_language.keys())]:
        print(f"{language}: {mean(shared_context_size_per_language[language]['left']):.2f} left, {mean(shared_context_size_per_language[language]['right']):.2f} right")
    print()

    print("Trigger point frequency:")
    for trigger, freq in sorted(triggerpoint_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"{trigger}: {freq}")
    print()

    print("IDE frequency:")
    for ide, freq in ide_lang_freq.items():
        for language in ["all", *set(languages).intersection(freq.keys())]:
            print(f"{ide} ({language}): {freq[language]}")
    print()

    print("Samples per model:")
    for model, freq in sorted(model_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {freq}")
    print()

    print("Model frequency per language:")
    for model, freq_per_language in model_lang_freq.items():
        print(f"{model}: {freq_per_language}")
    print()

    print("Language frequency:")
    for language in set(languages).intersection(lang_freq.keys()):
        print(f"{language}: {lang_freq[language]['valid']} valid, {lang_freq[language]['total']} total")
    print()

    print(f"Total empty predictions: {empty_preds_total}")
    print("Empty predictions per model per language:")
    for model, empty_preds_per_language in empty_preds_per_model_per_language.items():
        for language, emp in empty_preds_per_language.items():
            if language in languages:
                print(f"{model} ({language}): {emp} = {100 * emp / total_files:.2f}% of total files")
    print()

    print(f"Total empty gt: {empty_gt_total}")
    print("Empty gt per model per language:")
    for model, empty_gt_per_language in empty_gt_per_model_per_language.items():
        for language, emp in empty_gt_per_model_per_language.items():
            if language in languages:
                print(f"{model} ({language}): {emp} = {100 * emp / total_files:.2f}% of total files")
    print()

    print(f"Acceptance rate: {100 * accepted_prediction_count / valid_files:.2f}%")
    print(f"-> {accepted_prediction_count} accepted predictions out of {valid_files} valid files")
    print()
    for model in acceptance_per_model_per_language.keys():
        print(f"Acceptance rate per language for {model}")
        for language in ["all", *set(languages).intersection(acceptance_per_model_per_language[model].keys())]:
            if language not in acceptance_per_model_per_language[model]:
                print(f"Acceptance rate for {language}: ?%")
            print(f"Acceptance rate for {language}: {100 * acceptance_per_model_per_language[model][language]['accepted'] / acceptance_per_model_per_language[model][language]['total']:.2f}%")
        print()

    print("Acceptance rate per language:")
    for language in ["all", *set(languages).intersection(acceptance_per_language.keys())]:
        print(f"Acceptance rate for {language}: {100 * acceptance_per_language[language]['accepted'] / acceptance_per_language[language]['total']:.2f}%")
        print(f"-> {acceptance_per_language[language]['accepted']} accepted predictions out of {acceptance_per_language[language]['total']} valid predictions")
    print()

    chosen_langs_acceptance = sum(acceptance_per_language[language]['accepted'] for language in languages if language in acceptance_per_language) / sum(acceptance_per_language[language]['total'] for language in languages if language in acceptance_per_language)
    print(f"Acceptance rate for chosen languages: {100 * chosen_langs_acceptance:.2f}%")
    print()

    print("Creating plots")

    triggers_to_plot = sorted((k for k in triggerpoint_freq.keys() if k != 'all'), reverse=True, key=lambda k: triggerpoint_freq[k])[:14]

    create_model_language_plots(results_by_language, args, languages=languages)
    create_triggerpoint_plots(results_by_trigger, args, triggers=triggers_to_plot)
    create_model_language_plots(accepted_results_by_language, args, only_accepted=True, languages=languages)
    create_triggerpoint_plots(accepted_results_by_trigger, args, only_accepted=True, triggers=triggers_to_plot)
    create_acceptance_rate_plots(acceptance_per_model_per_language, args, languages=languages)

def mean(l):
    return sum(l) / len(l)

def translate_data(data):
    # translate old data format to new data format if needed

    is_new = 'modelPredictions' in data
    if is_new:
        return

    if 'model' in data:
        # data['model'] = 'modelname'
        # at first we provided one prediction, randomly selecting a model. now we provide all model predictions
        model_name = 'InCoder' if data['model'] == 'CodeFill' else data['model']  # CodeFill was briefly listed as a model, but InCoder was actually being used
        data['modelPredictions'] = {
            model_name: data['predictions']
        }
        del data['model']
    else:
        data['modelPredictions'] = {}


def is_valid(data):
    has_ground_truth = 'groundTruth' in data and data['groundTruth'] is not None and data['groundTruth'].strip() != ''
    has_predictions = 'predictions' in data and any(pred.strip() != '' for pred in data['predictions'])

    return has_ground_truth and has_predictions

def get_accepted_prediction(data):
    has_accepted_prediction = 'chosenPrediction' in data and data['chosenPrediction'] is not None and data['chosenPrediction'].strip() != ''
    if has_accepted_prediction:
        return data['chosenPrediction']

    return None

def get_models(data):
    return data['modelPredictions'].keys()

def get_model_predictions(data):
    preds = {}
    models = get_models(data)
    model_pred_dict = data['modelPredictions']
    for model in models:
        if model in model_pred_dict and len(model_pred_dict[model]) >= 1:
            preds[model] = model_pred_dict[model][0]
    return preds # { model: 'pred' }

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

lang_dict = {
    "python": "Python",
    "java": "Java",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "php": "PHP",
    "kotlin": "Kotlin",
    "rust": "Rust",
    "cpp": "C++",
    "go": "Go",
    "csharp": "C#",
    "c": "C",
    "scala": "Scala",
}
def fix_language(language: str):
    if language not in lang_dict:
        return language
    return lang_dict[language]

def fix_model(model: str):
    if model == "UniXCoder":
        return "UniXcoder"
    return model

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

def get_language(language):
    if language == 'python' or language == '.py' or language == 'py':
        return 'python'
    elif language == 'java' or language == '.java':
        return 'java'
    elif language == 'typescript' or language == '.ts' or language == 'ts':
        return 'typescript'
    elif language == 'php' or language == '.php':
        return 'php'
    elif language == 'vue':
        return 'vue'
    elif language == 'kotlin' or language == 'kt':
        return 'kotlin'
    elif language == 'typescriptreact' or language == '.tsx' or language == 'ts' or language == 'typescript jsx':
        return 'typescriptreact'
    elif language == 'javascript' or language == '.js' or language == 'js' or language == 'ecmascript 6':
        return 'javascript'
    elif language == 'robotframework':
        return 'robotframework'
    elif language == 'json' or language == '.json':
        return 'json'
    elif language == 'latex':
        return 'latex'
    elif language == 'html' or language == '.html':
        return 'html'
    elif language == 'javascriptreact' or language == '.jsx' or language == 'jsx':
        return 'javascriptreact'
    elif language == 'xml' or language == '.xml':
        return 'xml'
    elif language == 'go':
        return 'go'
    elif language == 'ruby':
        return 'ruby'
    elif language == 'csharp' or language == '.cs' or language == 'c#' or language == 'cs':
        return 'csharp'
    elif language == 'blade.php':
        return 'blade.php'
    elif language == 'markdown' or language == '.md' or language == 'md':
        return 'markdown'
    elif language == 'rust' or language == '.rs' or language == 'rs':
        return 'rust'
    elif language == 'css' or language == '.css' or language == 'scss':
        return 'css'
    elif language == 'objectivec':
        return 'objectivec'
    elif language == 'cpp' or language == '.cpp':
        return 'cpp'
    elif language == 'dart' or language == '.dart':
        return 'dart'
    elif language == 'sql' or language == '.sql':
        return 'sql'
    elif language == '.shellscript' or language == '.sh' or language == 'sh' or language == 'shellscript':
        return 'shellscript'
    elif language == 'prisma' or language == '.prisma':
        return 'prisma'
    elif language == 'yaml' or language == '.yaml' or language == 'yml' or language == '.yml':
        return 'yaml'
    elif language == 'txt' or language == '.txt' or language == 'text' or language == 'plaintext':
        return 'txt'
    elif language == 'swift' or language == '.swift':
        return 'swift'
    elif language == 'c' or language == '.c':
        return 'c'
    elif language == 'gitignore':
        return 'gitignore'
    elif language == 'groovy':
        return 'groovy'
    elif language == 'perl5':
        return 'perl5'
    elif language == 'less':
        return 'less'
    elif language == 'scala':
        return 'scala'
    elif language == 'julia':
        return 'julia'
    else:
        return 'other'

model_order = ['InCoder', 'UniXcoder', 'CodeGPT']

def create_model_language_plots(results_by_language, args, only_accepted=False, languages=None):
    if languages is None:
        languages = ["Python", "Java", "TypeScript", "JavaScript", "PHP", "Ruby", "Rust", "C++", "Go", "C#", "C", "Scala"]

    for plot_metric in args.metrics:
        models = sorted(results_by_language.keys(), key=lambda m: model_order.index(fix_model(m)))
        max_y = 0
        x = [i for i in range(len(languages))]
        width = 1 / (len(models) + 1)
        plt.ylabel(fix_metric(plot_metric))
        plt.xticks([i + width for i in x], [fix_language(l) for l in languages], rotation=45)
        plt.yticks([i for i in range(0, 101, 10)])
        plt.yticks([i + 5 for i in range(0, 101, 10)], minor=True)
        for model_idx, model in enumerate(models):
            for language_idx, language in enumerate(languages):
                unfixed_lang = next((lang for lang in results_by_language[model].keys() if fix_language(lang) == fix_language(language)), None)
                if unfixed_lang:
                    metric_values = results_by_language[model][unfixed_lang]
                    avg_metric_val = sum([m[plot_metric] for m in metric_values]) / len(metric_values)
                else:
                    avg_metric_val = 0
                plt.bar(x[language_idx] + model_idx * width, avg_metric_val, width * 0.75, color=colors[model_idx], label=fix_model(model), zorder=3)
                max_y = max(max_y, avg_metric_val)

        # plt.ylim(0, max_y + max(5, 0.125 * max_y))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=len(models))
        plt.tight_layout()
        plt.grid(axis='y', which='major', zorder=0, alpha=0.5, linestyle='-')
        plt.grid(axis='y', which='minor', zorder=0, alpha=0.5, linestyle='--')
        plt.savefig(os.path.join(args.plots_dir, f"languages{'-accepted' if only_accepted else ''}-{plot_metric}.svg"))
        plt.clf()

def create_triggerpoint_plots(results_by_trigger, args, only_accepted=False, triggers=None):
    if triggers is None:
        triggers = [".", "(", "=", ",", "/", "[", "-", "if", "return", "+", "<", ">", "*", "{"]

    for plot_metric in args.metrics:
        models = sorted(results_by_trigger.keys(), key=lambda m: model_order.index(fix_model(m)))
        max_y = 0
        x = [i for i in range(len(triggers))]
        width = 1 / (len(models) + 1)
        plt.ylabel(fix_metric(plot_metric))
        plt.xticks([i + width for i in x], triggers)
        plt.yticks([i for i in range(0, 101, 10)])
        plt.yticks([i + 5 for i in range(0, 101, 10)], minor=True)
        for model_idx, model in enumerate(models):
            for trigger_idx, trigger in enumerate(triggers):
                if trigger in results_by_trigger[model]:
                    metric_values = results_by_trigger[model][trigger]
                    avg_metric_val = sum([m[plot_metric] for m in metric_values]) / len(metric_values)
                else:
                    avg_metric_val = 0
                plt.bar(x[trigger_idx] + model_idx * width, avg_metric_val, width * 0.75, color=colors[model_idx], label=fix_model(model), zorder=3)
                max_y = max(max_y, avg_metric_val)

#         plt.ylim(0, max_y + max(5, 0.125 * max_y))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=len(models))
        plt.tight_layout()
        plt.grid(axis='y', which='major', zorder=0, alpha=0.5, linestyle='-')
        plt.grid(axis='y', which='minor', zorder=0, alpha=0.5, linestyle='--')
        plt.savefig(os.path.join(args.plots_dir, f"triggers{'-accepted' if only_accepted else ''}-{plot_metric}.svg"))
        plt.clf()

def create_acceptance_rate_plots(acceptance_per_model_per_language, args, languages=None):
    if languages is None:
        languages = ["Python", "Java", "TypeScript", "JavaScript", "PHP", "Ruby", "Rust", "C++", "Go", "C#", "C", "Scala"]

    models = sorted(acceptance_per_model_per_language.keys(), key=lambda m: model_order.index(fix_model(m)))
    max_y = 0
    x = [i for i in range(len(languages))]
    width = 1 / (len(models) + 1)
    plt.ylabel("Acceptance %")
    plt.xticks([i + width for i in x], [fix_language(l) for l in languages], rotation=45)
    plt.yticks([i for i in range(0, 11, 1)])
    plt.yticks([i + 0.5 for i in range(0, 11, 1)], minor=True)
    for model_idx, model in enumerate(models):
        for language_idx, language in enumerate(languages):
            unfixed_lang = next((lang for lang in acceptance_per_model_per_language[model].keys() if fix_language(lang) == fix_language(language)), None)
            accept_rate = 0
            if unfixed_lang:
                obj = acceptance_per_model_per_language[model][language]
                total, accepted = obj["total"], obj["accepted"]
                if total > 0:
                    accept_rate = accepted / total * 100
            plt.bar(x[language_idx] + model_idx * width, accept_rate, width * 0.75, color=colors[model_idx], label=fix_model(model), zorder=3)
            max_y = max(max_y, accept_rate)

#     plt.ylim(0, max_y + 1)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=len(models))
    plt.tight_layout()
    plt.grid(axis='y', which='major', zorder=0, alpha=0.5, linestyle='-')
    plt.grid(axis='y', which='minor', zorder=0, alpha=0.5, linestyle='--')
    plt.savefig(os.path.join(args.plots_dir, f"language-acceptance.svg"))
    plt.clf()


if __name__ == "__main__":
    main()
    
