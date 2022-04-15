"""
This script loads in a file which is the output of galactic dependencies dataset.
It converts it both to a monolingual corpus a synthetic language corpus.
"""

import argparse
from tqdm import tqdm
import os
import pandas
import json


def convert_to_document_mlm(args):
    # Store lines in the file
    lines = open(args.galactic_file, 'r').readlines()

    # Store both in the monolingual and synthetic language corpus
    monolingual = []
    synthetic = []

    # Parse the files
    for line in lines:
        # If line starts with `# sentence-tokens-src:` then it is monolingual corpus
        if line.startswith('# sentence-tokens-src:'):
            start_string = '# sentence-tokens-src:'
            monolingual.append(line[len(start_string):]+'\n')
        elif line.startswith('# sentence-tokens:'):
            start_string = '# sentence-tokens:'
            synthetic.append(line[len(start_string):]+'\n')

    # Locate file directory
    _, original_file_name = os.path.split(args.galactic_file)
    file_dir = args.file_dir

    # Store the monolingual file
    mono_file = os.path.join(file_dir, '{}_{}'.format('mono', original_file_name))
    f = open(mono_file, 'w')
    f.writelines(monolingual)
    f.close()

    # Store the synthetic file
    synthetic_file = os.path.join(file_dir, '{}_{}'.format('synthetic', original_file_name))
    f = open(synthetic_file, 'w')
    f.writelines(monolingual + synthetic)
    f.close()


def convert_conllu_to_document(args):
    # Check the task type
    if args.task == 'mlm':
        convert_to_document_mlm(args)      
    else:
        raise('No support for this task type: {}'.format(args.task))

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--galactic_file", type=str, default='en', help="File with galactic dependencies output")
    parser.add_argument("--file_dir", type=str, default=None, help="Folder to store file in")
    parser.add_argument("--task", default='mlm', type=str, help="mlm/mnli/xnli/..../")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    convert_conllu_to_document(args)

if __name__ == '__main__':
    main()