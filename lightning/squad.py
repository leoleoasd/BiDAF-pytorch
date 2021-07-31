import os
import torch
from torchtext.datasets import SQuAD1
from torchtext.vocab import build_vocab_from_iterator
import itertools
import json
import nltk


# standard SQuAD is:
# (C, Q, [A], start_index)

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

class SQuAD(torch.utils.data.IterableDataset):
    def __init__(self, data_path):
        def iterate():
            with open(data_path) as json_file:
                raw_json_data = json.load(json_file)['data']
                for layer1 in raw_json_data:
                    for layer2 in layer1['paragraphs']:
                        for layer3 in layer2['qas']:
                            _context, _question = layer2['context'], layer3['question']
                            _answers = [item['text'] for item in layer3['answers']]
                            _idx = layer3['id']
                            _answer_start = [item['answer_start'] for item in layer3['answers']]
                            if len(_answers) == 0:
                                _answers = [""]
                                _answer_start = [-1]
                            # yield the raw data in the order of id, record[1], question, answers, answer_start
                            yield _idx, _context, _question, _answers, _answer_start
        self.iter = iterate()

    def __iter__(self):
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']
        for record in self.iter:
            for index in range(len(record[3])):
                s_idx = record[4][index]
                e_idx = record[4][index] + len(record[3][index])
                tokens = word_tokenize(record[1])
                l = 0
                s_found = False
                # find e_idx
                for i, t in enumerate(tokens):
                    while l < len(record[1]):
                        if record[1][l] in abnormals:
                            l += 1
                        else:
                            break
                    # exceptional cases
                    if t[0] == '"' and record[1][l:l + 2] == '\'\'':
                        t = '\'\'' + t[1:]
                    elif t == '"' and record[1][l:l + 2] == '\'\'':
                        t = '\'\''

                    l += len(t)
                    if l > s_idx and s_found == False:
                        s_idx = i
                        s_found = True
                    if l >= e_idx:
                        e_idx = i
                        break
                c_word = list(word_tokenize(record[1]))
                q_word = list(word_tokenize(record[2]))
                yield dict([('id', record[0]),
                          ('c_word', c_word),
                          ('c_char', [list(i) for i in c_word]),
                          ('q_word', q_word),
                          ('q_char', [list(i) for i in q_word]),
                          ('question', record[2]),
                          ('answer', record[3][index]),
                          ('s_idx', s_idx),
                          ('e_idx', e_idx)])
