from copy import deepcopy
from iteration_utilities import split
import pdb

in_files = ["../../../bucket/pretrain_data/en/train.txt", "../../../bucket/pretrain_data/en/valid.txt"]
out_syn_files = ["../../../bucket/henry_invert_data/pretrain/en/train_inv.txt", "../../../bucket/henry_invert_data/pretrain/en/valid_inv.txt"]
out_orig_files = ["../../../bucket/henry_invert_data/pretrain/en/train_orig.txt", "../../../bucket/henry_invert_data/pretrain/en/valid_orig.txt"]

# in_files = ["../sample_data/valid.txt"]
# out_syn_files = ["../sample_data/valid_inv.txt"]
# out_orig_files = ["../sample_data/valid_orig.txt"]

# in_files = ["../../../bucket/pretrain_data/en/valid.txt"]
# out_syn_files = ["../../../bucket/henry_invert_data/pretrain/en/valid_inv.txt"]
# out_orig_files = ["../../../bucket/henry_invert_data/pretrain/en/valid_orig.txt"]

end_chars = ['.', '!', '?']

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

    # data = []
    for t in text:
        data = list(split(t.split(' '), lambda x: x in end_chars, keep_before=True))
        for d in data:
            ori_sentences.append(" ".join(d))
            inv_d = deepcopy(d)
            if inv_d[-1] in end_chars:
                inv_d[:-1] = inv_d[:-1][::-1]
            else:
                inv_d = inv_d[::-1]
            syn_sentences.append(" ".join(inv_d))

    with open(out_syn_file, 'w') as f:
        f.write('\n'.join(syn_sentences))

    with open(out_orig_file, 'w') as f:
        f.write('\n'.join(ori_sentences))