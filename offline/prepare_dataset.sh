#!/bin/bash

# Download the dataset if not already present
if [ ! -f "unseen_test_sets.tar.gz" ]; then
  wget https://zenodo.org/record/6363556/files/unseen_test_sets.tar.gz
fi
tar -xvzf unseen_test_sets.tar.gz
python3 deduplicate_unseen.py
