from copy import deepcopy
import json

input_files = ['../../../bucket/supervised_data/ner/en/train.json', '../../../bucket/supervised_data/ner/en/dev.json']
output_orig_files = ['../../../bucket/henry_invert_data/ner/en/train_en_orig.json', '../../../bucket/henry_invert_data/ner/en/dev_en_orig.json']
output_syn_files = ['../../../bucket/henry_invert_data/ner/en/train_en_inv.json', '../../../bucket/henry_invert_data/ner/en/dev_en_inv.json']

# input_files = ["../sample_data/ner_train.json"]
# output_syn_files = ["../sample_data/ner_syn.json"]
# output_orig_files = ["../sample_data/ner_orig.json"]

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
            syn_sentences[i]['ner_tags'][:-1] = syn_sentences[i]['ner_tags'][:-1][::-1]
        else:
            syn_sentences[i]['tokens'] = syn_sentences[i]['tokens'][::-1]
            syn_sentences[i]['ner_tags'] = syn_sentences[i]['ner_tags'][::-1]

        n = len(syn_sentences[i]['ner_tags'])
        prev_idx = 0
        flag = False # False -> not part of an entity, True -> part of a IIIB entity (must flip)
        for j in range(n):
            B = 'B' in syn_sentences[i]['ner_tags'][j]
            I = 'I' in syn_sentences[i]['ner_tags'][j]
            if not B and not I:
                flag = False
                prev_idx = j+1
            elif B and not flag: # Single tag word, just move on
                flag = False
                prev_idx = j+1
            elif B and flag: # IIIIB, flip and reset
                syn_sentences[i]['ner_tags'][prev_idx:j+1] = syn_sentences[i]['ner_tags'][prev_idx:j+1][::-1]
                flag = False
                prev_idx = j+1
            elif I: # Start or in the middle of IIIIB
                flag = True

    with open(output_orig_file, 'w') as f:
        for s in orig_sentences:
            json.dump(s, f)
            f.write('\n')
    
    with open(output_syn_file, 'w') as f:
        for s in syn_sentences:
            json.dump(s, f)
            f.write('\n')

