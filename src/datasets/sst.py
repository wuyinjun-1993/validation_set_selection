import os
import texar.torch as tx
import tensorflow as tf
import re
import random
import torchtext
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import logging
MAX_SEQ_LENGTH = 64


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

    def __getitem__(self, item):
        return [self.input_ids, self.input_mask,
                self.segment_ids, self.label_id][item]


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then each
    # token that's truncated likely contains more information than a longer
    # sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

    return features


class DatasetProcessor:
    def get_train_examples(self):
        raise NotImplementedError

    def get_dev_examples(self):
        raise NotImplementedError

    def get_test_examples(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

def clean_sst_text(text):
    """Cleans tokens in the SST data, which has already been tokenized.
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

def transform_raw_sst(data_path, raw_fn, new_fn):
    """Transforms the raw data format to a new format.
    """
    fout_x_name = os.path.join(data_path, new_fn + '.sentences.txt')
    fout_x = open(fout_x_name, 'w', encoding='utf-8')
    fout_y_name = os.path.join(data_path, new_fn + '.labels.txt')
    fout_y = open(fout_y_name, 'w', encoding='utf-8')

    fin_name = os.path.join(data_path, raw_fn)
    with open(fin_name, 'r', encoding='utf-8') as fin:
        for line in fin:
            parts = line.strip().split()
            label = parts[0]
            sent = ' '.join(parts[1:])
            sent = clean_sst_text(sent)
            fout_x.write(sent + '\n')
            fout_y.write(label + '\n')

    return fout_x_name, fout_y_name
def prepare_data(data_path):
    """Preprocesses SST2 data.
    """
    train_path = os.path.join(data_path, "sst.train.sentences.txt")
    # if not tf.gfile.Exists(train_path):
    if not os.path.exists(train_path):
        url = ('https://raw.githubusercontent.com/ZhitingHu/'
               'logicnn/master/data/raw/')
        files = ['stsa.binary.phrases.train', 'stsa.binary.dev',
                 'stsa.binary.test']
        for fn in files:
            tx.data.maybe_download(url + fn, data_path, extract=True)

    fn_train, _ = transform_raw_sst(
        data_path, 'stsa.binary.phrases.train', 'sst2.train')
    transform_raw_sst(data_path, 'stsa.binary.dev', 'sst2.dev')
    transform_raw_sst(data_path, 'stsa.binary.test', 'sst2.test')

    vocab = tx.data.make_vocab(fn_train)
    fn_vocab = os.path.join(data_path, 'sst2.vocab')
    with open(fn_vocab, 'w', encoding='utf-8') as f_vocab:
        for v in vocab:
            f_vocab.write(v + '\n')

    logging.info('Preprocessing done: {}'.format(data_path))


def _subsample_by_classes(all_examples, labels, num_per_class=None):
    if num_per_class is None:
        return all_examples

    examples = {label: [] for label in labels}
    for example in all_examples:
        examples[example.label].append(example)

    selected_examples = []
    for label in labels:
        random.shuffle(examples[label])

        num_in_class = num_per_class[label]
        selected_examples = selected_examples + examples[label][:num_in_class]
        print('number of examples with label \'{}\': {}'.format(
            label, num_in_class))

    return selected_examples


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def get_bert_optimizer(model, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate)

class sst_dataset(Dataset):
    def __init__(self, all_input_ids, all_input_mask, all_segment_ids, all_label_ids):

        # super(new_mnist_dataset, self).__init__(*args, **kwargs)

        self.all_input_ids, self.all_input_mask, self.all_segment_ids, self.targets = all_input_ids, all_input_mask, all_segment_ids, all_label_ids

    def __getitem__(self, index):
        input_ids, input_mask, segment_ids = self.all_input_ids[index], self.all_input_mask[index], self.all_segment_ids[index]
        target = self.targets[index]
        # img, target = self.data[index], self.targets[index]
        
        return (index, (input_ids, input_mask, segment_ids), target)
        # image, target = super(new_mnist_dataset, self).__getitem__(index)

        # return (index, image,target)

    def __len__(self):
        return len(self.all_input_ids)

    @staticmethod
    def get_subset_dataset(dataset, sample_ids, labels = None):
        input_ids, input_mask, segment_ids, label_ids = dataset.all_input_ids[sample_ids].clone(), dataset.all_input_mask[sample_ids].clone(), dataset.all_segment_ids[sample_ids].clone(), dataset.targets[sample_ids].clone()
        
        if labels is None:
            subset_labels = label_ids
        else:
            subset_labels = labels[sample_ids].clone()

        return sst_dataset(input_ids, input_mask, segment_ids, subset_labels)


    @staticmethod
    def concat_validset(dataset1, dataset2):
        input_ids1, input_mask1, segment_ids1, targets1 = dataset1.all_input_ids, dataset1.all_input_mask, dataset1.all_segment_ids, dataset1.targets
        input_ids2, input_mask2, segment_ids2, targets2 = dataset2.all_input_ids, dataset2.all_input_mask, dataset2.all_segment_ids, dataset2.targets

        all_input_ids = torch.cat([input_ids1, input_ids2], dim = 0)
        all_input_mask = torch.cat([input_mask1, input_mask2], dim = 0)
        all_segment_ids = torch.cat([segment_ids1, segment_ids2], dim=0)
        all_targets = torch.cat([targets1, targets2], dim = 0)

        
        return sst_dataset(all_input_ids, all_input_mask, all_segment_ids, all_targets)
    
    @staticmethod
    def to_cuda(data, targets):
        input_ids, input_mask, segment_ids = data
        
        return (input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()), targets.cuda()

class SST5Processor(DatasetProcessor):
    """Processor for the SST-5 data set."""

    def __init__(self):
        TEXT = torchtext.data.Field()
        LABEL = torchtext.data.Field(sequential=False)

        self._train_set, self._dev_set, self._test_set = \
            torchtext.datasets.SST.splits(
                TEXT, LABEL, fine_grained=True)

    def get_train_examples(self, num_per_class=None, noise_rate=0.):
        """See base class."""
        print('getting train examples...')
        all_examples = self._create_examples(self._train_set, "train")

        # Add noise
        for i, _ in enumerate(all_examples):
            if random.random() < noise_rate:
                all_examples[i].label = random.choice(self.get_labels())

        return _subsample_by_classes(
            all_examples, self.get_labels(), num_per_class)

    def get_dev_examples(self, num_per_class=None):
        """See base class."""
        print('getting dev examples...')
        all_examples = self._create_examples(self._dev_set, "dev")

        return _subsample_by_classes(
            all_examples, self.get_labels(), num_per_class)

    def get_test_examples(self):
        """See base class."""
        print('getting test examples...')
        return self._create_examples(self._test_set, "test")

    def get_labels(self):
        """See base class."""
        return ['negative', 'very positive', 'neutral',
                'positive', 'very negative']

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

def create_train_valid_test_set(sstprocess, label_list, tokenizer):
    train_examples = sstprocess.get_train_examples()
    valid_examples = sstprocess.get_dev_examples()
    test_examples = sstprocess.get_test_examples()
    train_dataset = sst_make_dataset(train_examples, label_list, tokenizer)
    valid_dataset = sst_make_dataset(valid_examples, label_list, tokenizer)
    test_dataset = sst_make_dataset(test_examples, label_list, tokenizer)
    return train_dataset, valid_dataset, test_dataset

def sst_make_dataset(examples, label_list, tokenizer):
    all_features = _convert_examples_to_features(
        examples=examples,
        label_list=label_list,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        output_mode='classification')

    all_input_ids = torch.tensor(
        [f.input_ids for f in all_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in all_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in all_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in all_features], dtype=torch.long)
    # all_ids = torch.arange(len(examples))

    # dataset = TensorDataset(
    #     all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ids)

    dataset = sst_dataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset

class SST2Processor(DatasetProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, data_path):
        self.data_path = data_path
        prepare_data(data_path)

    def get_train_examples(self, num_per_class=None, noise_rate=0.):
        print('getting train examples...')
        all_examples = self._create_examples("train")

        # Add noise
        for i, _ in enumerate(all_examples):
            if random.random() < noise_rate:
                all_examples[i].label = random.choice(self.get_labels())

        return _subsample_by_classes(
            all_examples, self.get_labels(), num_per_class)

    def get_dev_examples(self, num_per_class=None):
        print('getting dev examples...')
        return _subsample_by_classes(
            self._create_examples("dev"), self.get_labels(), num_per_class)

    def get_test_examples(self):
        print('getting test examples...')
        return self._create_examples("test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, set_type):
        """Creates examples for the training and dev sets."""
        sentence_file = open(self.data_path + '/sst2.{}.sentences.txt'.format(set_type))
        labels_file = open(self.data_path + '/sst2.{}.labels.txt'.format(set_type))

        examples = []
        for sentence, label in zip(
                sentence_file.readlines(), labels_file.readlines()):
            label = label.strip('\n')
            sentence = sentence.strip('\n')

            if label == '':
                break
            examples.append(InputExample(
                guid=set_type, text_a=sentence, text_b=None, label=label))
        return examples