from copy import deepcopy
import json
import pdb

input_files = ['../../bucket/supervised_data/ner/en/dep/synthetic_dep_flattened_train-en~fr@N~fr@V.json', '../../bucket/supervised_data/ner/en/dep/synthetic_dep_flattened_dev-en~fr@N~fr@V.json']
output_orig_files = ['../../bucket/henry_syntax_data/ner/en/train_en_fr_orig.json', '../../bucket/henry_syntax_data/ner/en/dev_en_fr_orig.json']
output_syn_files = ['../../bucket/henry_syntax_data/ner/en/train_en_fr_synt.json', '../../bucket/henry_syntax_data/ner/en/dev_en_fr_synt.json']

# input_files = ["../sample_data/ner_train.json"]
# output_syn_files = ["../sample_data/ner_syn.json"]
# output_orig_files = ["../sample_data/ner_orig.json"]

label_to_id = {'B-LOC': 0, 'B-ORG': 1, 'B-PER': 2, 'I-LOC': 3, 'I-ORG': 4, 'I-PER': 5, 'O': 6}
inv_label_to_id = {v: k for k, v in label_to_id.items()}
B_labels = [key for key in label_to_id.keys() if 'B' in key]
I_labels = [key for key in label_to_id.keys() if 'I' in key]
B_to_I = dict([('B-'+suffix.split('-')[-1], 'I-'+suffix.split('-')[-1]) for suffix in label_to_id.keys() if 'B' in suffix])
I_to_B = dict([('I-'+suffix.split('-')[-1], 'B-'+suffix.split('-')[-1]) for suffix in label_to_id.keys() if 'B' in suffix])

def make_consistent(example):
    n = len(example)
    flag = False
    prev_suffix = None
    for j in range(n):
        # if example[j] == 'O':
        #     continue
        if example[j] not in I_labels and example[j] not in B_labels:
            flag = False
            prev_suffix = None
        elif not flag and example[j] in I_labels:
            example[j] = I_to_B[example[j]]
            flag = True
            prev_suffix = example[j].split('-')[-1]
        elif not flag and example[j] in B_labels:
            flag = True
            prev_suffix = example[j].split('-')[-1]
        elif flag and example[j] in B_labels and prev_suffix == example[j].split('-')[-1]:
            example[j] = B_to_I[example[j]]
        elif flag and example[j] in B_labels and prev_suffix != example[j].split('-')[-1]:
            prev_suffix = example[j].split('-')[-1]
        elif flag and example[j] in I_labels and prev_suffix != example[j].split('-')[-1]:
            example[j] = I_to_B[example[j]]
            prev_suffix = example[j].split('-')[-1]                    
    return example

for i in range(len(input_files)):
    input_file = input_files[i]
    output_orig_file = output_orig_files[i]
    output_syn_file = output_syn_files[i]

    orig_sentences = []
    print("Started Reading JSON file which contains multiple JSON document")
    with open(input_file) as f:
        for jsonObj in f:
            orig_sentences.append(json.loads(jsonObj))

    print("Printing each JSON Decoded Object")
    end_tokens = ['.', '?', '!']

    syn_sentences = deepcopy(orig_sentences)
    for i in range(len(syn_sentences)):
        make_consistent(syn_sentences[i]['ner_tags'])

    with open(output_orig_file, 'w') as f:
        for s in orig_sentences:
            json.dump(s, f)
            f.write('\n')
    
    with open(output_syn_file, 'w') as f:
        for s in syn_sentences:
            json.dump(s, f)
            f.write('\n')

