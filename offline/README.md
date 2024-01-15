# Offline Evaluation

## Replication
This directory contains the code for the analysis of code completion models on artificially created code completion tasks from the Unseen dataset.
Unseen can be retrieved as per the instructions [here](https://github.com/VHellendoorn/Code-LMs#evaluation):
We also provide a script to do this automatically:
```shell
./prepare_dataset.sh
```
This script also de-duplicates the Unseen dataset against the training set of the CodeSearchNet dataset, and the repositories that InCoder was trained on.
This makes the Unseen dataset usable for evaluation of InCoder, UniXcoder, and our own CodeGPT model trained on CodeSearchNet.

A list of repositories in the CodeSearchNet dataset is already provided in the `codesearchnet_repos.txt` file.
However, if you wish to generate this file yourself you can do so by running
```shell
python get_csn_repos.py
```
Note that this will download the CodeSearchNet dataset, which will take up roughly 6GB of disk space.

An analysis of the contents of the de-duplicated Unseen dataset can be created by running.
This will generate a LaTeX table.
```shell
python analyze_unseen.py
```

To create test sets with code completion tasks from the Unseen dataset, run the following:
```shell
python create_test_set_unseen.py -m random
python create_test_set_unseen.py -m triggerpoint
```
This will create code completion tasks 1) randomly, and 2) using trigger points.
By default, they will be placed in the `output-unseen` directory.

Then, run the models using `models/run_models.py`
E.g.:
```shell
python models/run_models.py output-unseen unixcoder incoder codegpt_csn
```
For CodeGPT the checkpoint trained on CodeSearchNet is required.
We include it in the `models/codegpt-csn-checkpoint` directory.
We temporarily use Git LFS to store this checkpoint as uploading to HuggingFace would reveal identifiable information.
The models will place their predictions in the `output-unseen/predictions` directory.

Finally, plots can be generated with the following command:
```shell
python evaluate.py
```
Before running this command, make sure that the `output-unseen` directory contains the predictions of the models.
Plots are placed in the `plots` directory.
To print LaTeX tables, use the `--latex` flag.

## output-unseen.zip
If you only want to run evaluation, you can unzip the `output-unseen.zip` file and run the evaluation script.
This `.zip` contains the test sets used in our paper, along with the predictions of the models.

## Plots
The `plots` directory contains all plots used in the paper, along with plots for the metrics that are not included in the paper.
