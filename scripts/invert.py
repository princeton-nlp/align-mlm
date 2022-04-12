import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
# nltk.download('averaged_perceptron_tagger')

# with open("MultilingualAnalysis/sample_data/train.txt") as f:
#     for line in f:
#         process_lines(line)
file1 = open("MultilingualAnalysis/sample_data/train.txt","r+")
text = file1.readlines()
file1.close()

n = len(text)
for i in range(n):
    text[i] = text[i].strip()

text = list(filter(None, text))
sentences = []

for t in text:
    sent_text = nltk.sent_tokenize(t) # this gives us a list of sentences
    # now loop over each sentence and tokenize it separately
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)
        if (tokenized_text[-1] == '.' or tokenized_text[-1] == '?' or tokenized_text[-1] == '!'):
            tokenized_text[:-1] = tokenized_text[:-1][::-1]
        else:
            tokenized_text = tokenized_text[::-1]
        sentences.append(TreebankWordDetokenizer().detokenize(tokenized_text))

# for s in sentences:
#     print(s)

with open('inverted.txt', 'w') as f:
    # for item in sentences:
    #     f.write("%s\n" % item)
    f.write('\n'.join(sentences))