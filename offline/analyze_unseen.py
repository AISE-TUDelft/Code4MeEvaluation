"""
Evaluates the unseen dataset: gets the LOC, number of files, and number of repos.
Includes both totals and per-language breakdowns.
"""
import os

UNSEEN_PATH = 'Code-sampled100'

def main():
    if not os.path.exists(UNSEEN_PATH):
        print('Code-sampled100 not found')
        exit(1)

    languages = set()
    repos = set()
    repos_per_language = {}
    total_loc = 0
    loc_per_language = {}
    total_files = 0
    files_per_language = {}

    for language in os.listdir(UNSEEN_PATH):
        languages.add(language)
        repos_per_language[language] = set()
        loc_per_language[language] = 0
        files_per_language[language] = 0
        for user in os.listdir(os.path.join(UNSEEN_PATH, language)):
            for repo in os.listdir(os.path.join(UNSEEN_PATH, language, user)):
                full_repo_name = f'{user}/{repo}'
                repos.add(full_repo_name)
                repos_per_language[language].add(full_repo_name)
                for file in os.listdir(os.path.join(UNSEEN_PATH, language, user, repo)):
                    total_files += 1
                    files_per_language[language] += 1
                    with open(os.path.join(UNSEEN_PATH, language, user, repo, file), 'r') as f:
                        loc = sum(1 for _ in f)
                        total_loc += loc
                        loc_per_language[language] += loc


    column_names = ["Language", "Repos", "Files", "LOC"]
    column_widths = [10, 15, 15, 20]
    column_alignments = ['<', '>', '>', '>']
    header = ''
    for column_name, column_alignment, column_width in zip(column_names, column_alignments, column_widths):
        header += f"{column_name: {column_alignment}{column_width}} & "
    print(header.strip().rstrip("&") + "\\\\")
    for language in sorted(languages):
        row = [
            language,
            f'\\numprint{"{"}{len(repos_per_language[language])}{"}"}',
            f'\\numprint{"{"}{files_per_language[language]}{"}"}',
            f'\\numprint{"{"}{loc_per_language[language]}{"}"}',
        ]
        row_str = ''
        for column, column_alignment, column_width in zip(row, column_alignments, column_widths):
            row_str += f"{column: {column_alignment}{column_width}} & "
        print(row_str.strip().rstrip("&") + "\\\\")

    print("-" * 30)
    print(f"Total\t\t{len(repos)} repos\t{total_files} files\t{total_loc} LOC")




if __name__ == '__main__':
    main()
