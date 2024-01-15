import os
from shutil import rmtree

REPO_LIST_FILES = [
    'incoder_repos.txt',
    'codesearchnet_repos.txt',
]

for repo_list_file in REPO_LIST_FILES:
    if not os.path.exists(repo_list_file):
        print(f'{repo_list_file} not found')
        exit(1)

UNSEEN_PATH = 'Code-sampled100'

if not os.path.exists(UNSEEN_PATH):
    print('Code-sampled100 not found')
    exit(1)

train_repo_set = set()

for repo_list_file in REPO_LIST_FILES:
    repo_count = 0
    with open(repo_list_file, 'r') as f:
        for line in f:
            train_repo_set.add(line.strip())
            repo_count += 1

    print(f'Total repos in {repo_list_file}: {repo_count}')


print(f'Total repos in train set: {len(train_repo_set)}')

total_removed = 0
total_repos = 0
for language in os.listdir(UNSEEN_PATH):
    print(f'Processing {language}...')
    language_path = os.path.join(UNSEEN_PATH, language)
    for user in os.listdir(language_path):
        for repo in os.listdir(os.path.join(language_path, user)):
            full_repo_name = f'{user}/{repo}'
            total_repos += 1
            if full_repo_name in train_repo_set:
                print(f'Removing {full_repo_name}')
                rmtree(os.path.join(language_path, user, repo))
                total_removed += 1

print(f'Total removed: {total_removed} / {total_repos}')
