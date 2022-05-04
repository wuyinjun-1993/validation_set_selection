import os
import texar.torch as tx
# import tensorflow as tf
import re
import random
import torchtext
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import logging
MAX_SEQ_LENGTH = 64

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets.sst import *

class imdb_Processor(DatasetProcessor):
    """Processor for the imdb data set."""
    # def __init__(self, data_path):
    #     self.data_path = data_path
    #     prepare_data(data_path)
    def __init__(self, data_path):
        # TEXT = torchtext.legacy.data.Field()
        # LABEL = torchtext.legacy.data.Field(sequential=False)
        TEXT = torchtext.legacy.data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm')
        LABEL = torchtext.legacy.data.LabelField(dtype = torch.float)
        self.TEXT = TEXT
        self.LABEL = LABEL
        self.data_path = data_path
        self._train_set, self._test_set = \
            torchtext.legacy.datasets.IMDB.splits(
                TEXT, LABEL, root=data_path)
        self._train_set, self._dev_set = self._train_set.split(random_state = random.seed(1), split_ratio=0.9)

    def get_train_examples(self, num_per_class=None, noise_rate=0.):
        """See base class."""
        print('getting train examples...')
        all_examples = self._create_examples(self._train_set, "train")

        # Add noise
        for i, _ in enumerate(all_examples):
            if random.random() < noise_rate:
                all_examples[i].label = random.choice(self.get_labels())
        return all_examples
        # return _subsample_by_classes(
        #     all_examples, self.get_labels(), num_per_class)

    def get_dev_examples(self, num_per_class=None):
        """See base class."""
        print('getting dev examples...')
        all_examples = self._create_examples(self._dev_set, "dev")
        return all_examples
        # return _subsample_by_classes(
        #     all_examples, self.get_labels(), num_per_class)

    def get_test_examples(self):
        """See base class."""
        print('getting test examples...')
        return self._create_examples(self._test_set, "test")

    def get_labels(self):
        """See base class."""
        return ['neg','pos']

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(
                guid=guid,
                text_a=' '.join(data.text),
                text_b=None,
                label=data.label))
        return examples


# def imdb_make_dataset(examples, label_list, tokenizer):
#     all_features = _convert_examples_to_features(
#         examples=examples,
#         label_list=label_list,
#         max_seq_length=MAX_SEQ_LENGTH,
#         tokenizer=tokenizer,
#         output_mode='classification')

#     all_input_ids = torch.tensor(
#         [f.input_ids for f in all_features], dtype=torch.long)
#     all_input_mask = torch.tensor(
#         [f.input_mask for f in all_features], dtype=torch.long)
#     all_segment_ids = torch.tensor(
#         [f.segment_ids for f in all_features], dtype=torch.long)
#     all_label_ids = torch.tensor(
#         [f.label_id for f in all_features], dtype=torch.long)


# def create_train_valid_test_set_imdb(sstprocess, label_list, tokenizer):
#     train_data = sstprocess.get_train_examples()
#     valid_data = sstprocess.get_dev_examples()
#     test_data = sstprocess.get_test_examples()

#     MAX_VOCAB_SIZE = 25_000

#     sstprocess.TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
#     sstprocess.LABEL.build_vocab(train_data)

#     # train_dataset = sst_make_dataset(train_examples, label_list, tokenizer)
#     # valid_dataset = sst_make_dataset(valid_examples, label_list, tokenizer)
#     # test_dataset = sst_make_dataset(test_examples, label_list, tokenizer)
#     return train_data, valid_data, test_data