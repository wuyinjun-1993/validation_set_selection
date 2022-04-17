import os
import texar as tx
import tensorflow as tf
import re
import random

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

    return fout_x_name, 
def prepare_data(data_path):
    """Preprocesses SST2 data.
    """
    train_path = os.path.join(data_path, "sst.train.sentences.txt")
    if not tf.gfile.Exists(train_path):
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

    tf.logging.info('Preprocessing done: {}'.format(data_path))


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

class SST2Processor(DatasetProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, data_path):
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
        sentence_file = open('data/sst2.{}.sentences.txt'.format(set_type))
        labels_file = open('data/sst2.{}.labels.txt'.format(set_type))

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