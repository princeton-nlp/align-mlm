import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from copy import deepcopy
nltk.download('punkt')

# with open("MultilingualAnalysis/sample_data/train.txt") as f:
#     for line in f:
#         process_lines(line)
in_files = ["../../bucket/pretrain_data/en/train.txt", "../../bucket/pretrain_data/en/valid.txt"]
out_syn_files = ["../../bucket/henry_invert_data/pretrain/en/train_inv.txt", "../../bucket/henry_invert_data/pretrain/en/valid_inv.txt"]
out_orig_files = ["../../bucket/henry_invert_data/pretrain/en/train_orig.txt", "../../bucket/henry_invert_data/pretrain/en/valid_orig.txt"]

# in_files = ["../../bucket/pretrain_data/en/valid.txt"]
# out_syn_files = ["../../bucket/henry_invert_data/pretrain/en/valid_inv.txt"]
# out_orig_files = ["../../bucket/henry_invert_data/pretrain/en/valid_orig.txt"]

for i in range(len(in_files)):
    in_file = in_files[i]
    out_syn_file = out_syn_files[i]
    out_orig_file = out_orig_files[i]

    file1 = open(in_file,"r+")
    text = file1.readlines()
    file1.close()

    n = len(text)
    for i in range(n):
        text[i] = text[i].strip()

    # text = list(filter(None, text))
    ori_sentences = []
    syn_sentences = []

    for t in text:
        sent_text = nltk.sent_tokenize(t) # this gives us a list of sentences
        # now loop over each sentence and tokenize it separately
        if t == '':
            syn_sentences.append('')
            ori_sentences.append('')
        for sentence in sent_text:
            tokenized_text = nltk.word_tokenize(sentence)
            ori_tokenized_text = deepcopy(tokenized_text)
            if (tokenized_text[-1] == '.' or tokenized_text[-1] == '?' or tokenized_text[-1] == '!'):
                tokenized_text[:-1] = tokenized_text[:-1][::-1]
            else:
                tokenized_text = tokenized_text[::-1]
            syn_sentences.append(TreebankWordDetokenizer().detokenize(tokenized_text))
            ori_sentences.append(TreebankWordDetokenizer().detokenize(ori_tokenized_text))

    # for s in sentences:
    #     print(s)

    with open(out_syn_file, 'w') as f:
        f.write('\n'.join(syn_sentences))

    with open(out_orig_file, 'w') as f:
        f.write('\n'.join(ori_sentences))