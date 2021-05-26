import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from tqdm import tqdm


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD():
    def __init__(self, args):
        path = '.data/squad'
        train_examples_paths = []
        dev_examples_paths = []
        test_examples_paths = []
        for i in args.dev_files:
            dataset_path = path + '/torchtext/' + i.replace("/", "_") + "/"
            train_examples_paths.append(dataset_path + 'train_examples.pt')
            dev_examples_paths.append(dataset_path + 'dev_examples.pt')
            test_examples_paths.append(dataset_path + 'test_examples.pt')


        print("preprocessing data files...")
        for i in args.dev_files:
            if not os.path.exists('{}/{}l'.format(path, i)):
                self.preprocess_file('{}/{}'.format(path, i))
        for i in args.train_files:
            if not os.path.exists('{}/{}l'.format(path, i)):
                self.preprocess_file('{}/{}'.format(path, i))
        for i in args.test_files:
            if not os.path.exists('{}/{}l'.format(path, i)):
                self.preprocess_file('{}/{}'.format(path, i))

        self.RAW = data.RawField()
        # explicit declaration for torchtext compatibility
        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]

        self.train = []
        self.dev = []
        self.test = []

        if all([os.path.exists(i) for i in train_examples_paths + dev_examples_paths + test_examples_paths]):
            print("loading splits...")
            for i in train_examples_paths:
                examples = torch.load(i)
                print(i, ":", len(examples))
                self.train.append(data.Dataset(examples=examples, fields=list_fields))
            for i in dev_examples_paths:
                examples = torch.load(i)
                print(i, ":", len(examples))
                self.dev.append(data.Dataset(examples=examples, fields=list_fields))
            for i in test_examples_paths:
                examples = torch.load(i)
                print(i, ":", len(examples))
                self.test.append(data.Dataset(examples=examples, fields=list_fields))
        else:
            print("building splits...")
            for train_path, dev_path, test_path, i in zip(args.train_files, args.dev_files, args.test_files, range(0, len(args.train_files))):
                train, dev, test  = data.TabularDataset.splits(
                    path=path,
                    train='{}l'.format(train_path),
                    validation='{}l'.format(dev_path),
                    test='{}l'.format(test_path),
                    format='json',
                    fields=dict_fields)

                try:
                    os.makedirs("".join(os.path.split(train_examples_paths[i])[:-1]))
                except FileExistsError:
                    pass
                torch.save(train.examples, train_examples_paths[i])
                torch.save(dev.examples, dev_examples_paths[i])
                torch.save(test.examples, test_examples_paths[i])
                self.train.append(train)
                self.dev.append(dev)
                self.test.append(test)

        #cut too long context in the training set for efficiency.
        if args.context_threshold > 0:
            for i in range(0, len(self.train)):
                print(len(self.train[i].examples))
                self.train[i].examples = [e for e in self.train[i].examples if len(e.c_word) <= args.context_threshold]
                print(len(self.train[i].examples))
            # self.other_train.examples = [e for e in self.other_train.examples if len(e.c_word) <= args.context_threshold]

        print("building vocab...")
        self.CHAR.build_vocab(*self.train, *self.dev, *self.test)
        self.WORD.build_vocab(*self.train, *self.dev, *self.test, vectors=GloVe(name='6B', dim=args.word_dim))
        print("CHAR SIZE", len(self.CHAR.vocab))
        print("WORD SIZE", len(self.WORD.vocab))
        print("building iterators...")
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        def train_bucket_iter(train):
            for i in range(0, len(train)):
                yield data.BucketIterator(
                    train.pop(0),
                    batch_size=args.train_batch_size,
                    device=device,
                    repeat=True,
                    shuffle=True,
                    sort_key=lambda x: len(x.c_word)
                )

        self.train_iter = train_bucket_iter(self.train)

        def dev_bucket_iter(dev):
            for i in range(0, len(dev)):
                yield data.BucketIterator(
                    dev.pop(0),
                    batch_size=args.dev_batch_size,
                    device=device,
                    repeat=False,
                    sort_key=lambda x: len(x.c_word)
                )

        self.dev_iter = dev_bucket_iter(self.dev)

        def test_bucket_iter(test):
            for i in range(0, len(test)):
                yield data.BucketIterator(
                    test.pop(0),
                    batch_size=args.test_batch_size,
                    device=device,
                    repeat=False,
                    sort_key=lambda x: len(x.c_word)
                )

        self.test_iter = test_bucket_iter(self.test)

    def preprocess_file(self, path):
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in tqdm(article['paragraphs']):
                    context = paragraph['context']
                    tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))

        with open('{}l'.format(path), 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)
