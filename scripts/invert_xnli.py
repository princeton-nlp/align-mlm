from copy import deepcopy
import json
import pdb
from iteration_utilities import split

input_files = ['../../bucket/supervised_data/xnli/en/train_en.json', '../../bucket/supervised_data/xnli/en/dev_en.json']
output_orig_files = ['../../bucket/henry_invert_data/xnli/en/train_en_orig.json', '../../bucket/henry_invert_data/xnli/en/dev_en_orig.json']
output_syn_files = ['../../bucket/henry_invert_data/xnli/en/train_en_inv.json', '../../bucket/henry_invert_data/xnli/en/dev_en_inv.json']

# input_files = ["../sample_data/xnli_train.json"]
# output_syn_files = ["../sample_data/xnli_syn.json"]
# output_orig_files = ["../sample_data/xnli_orig.json"]

end_chars = ['.', '!', '?']

def invert(sentence):
    # tokenized_text = nltk.word_tokenize(sentence)
    # # pdb.set_trace()
    # if (tokenized_text[-1] == '.' or tokenized_text[-1] == '?' or tokenized_text[-1] == '!'):
    #     tokenized_text[:-1] = tokenized_text[:-1][::-1]
    # else:
    #     tokenized_text = tokenized_text[::-1]
    # return TreebankWordDetokenizer().detokenize(tokenized_text)
    if (sentence == ''):
        return sentence
    data = sentence.split(' ')
    if data[-1][-1] in end_chars:
        # print(data[-1][-1])
        data.append(data[-1][-1])
        data[-2] = data[-2][:-1]
    if data[-1] in end_chars:
        data[:-1] = data[:-1][::-1]
    else:
        data = data[::-1]
    res = " ".join(data[:-1])
    if data[-1] in end_chars:
        res = f"{res}{data[-1]}"
    else:
        res = f"{res} {data[-1]}"
    return res

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

    # pdb.set_trace()
    syn_sentences = deepcopy(orig_sentences)
    # pdb.set_trace()
    for i in range(len(syn_sentences)):
        syn_sentences[i]['sentence1'] = invert(syn_sentences[i]['sentence1'])
        syn_sentences[i]['sentence2'] = invert(syn_sentences[i]['sentence2'])

    with open(output_orig_file, 'w') as f:
        for s in orig_sentences:
            json.dump(s, f)
            f.write('\n')
    
    with open(output_syn_file, 'w') as f:
        for s in syn_sentences:
            json.dump(s, f)
            f.write('\n')

