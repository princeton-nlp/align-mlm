from copy import deepcopy
import json

input_files = ['../../../bucket/supervised_data/pos/en/train-en.json', '../../../bucket/supervised_data/pos/en/dev-en.json']
output_orig_files = ['../../../bucket/henry_invert_data/pos/en/train_en_orig.json', '../../../bucket/henry_invert_data/pos/en/dev_en_orig.json']
output_syn_files = ['../../../bucket/henry_invert_data/pos/en/train_en_inv.json', '../../../bucket/henry_invert_data/pos/en/dev_en_inv.json']

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
        if (syn_sentences[i]['tokens'][-1] in end_tokens):
            syn_sentences[i]['tokens'][:-1] = syn_sentences[i]['tokens'][:-1][::-1]
            syn_sentences[i]['pos_tags'][:-1] = syn_sentences[i]['pos_tags'][:-1][::-1]
        else:
            syn_sentences[i]['tokens'] = syn_sentences[i]['tokens'][::-1]
            syn_sentences[i]['pos_tags'] = syn_sentences[i]['pos_tags'][::-1]

    with open(output_orig_file, 'w') as f:
        for s in orig_sentences:
            json.dump(s, f)
            f.write('\n')
    
    with open(output_syn_file, 'w') as f:
        for s in syn_sentences:
            json.dump(s, f)
            f.write('\n')

