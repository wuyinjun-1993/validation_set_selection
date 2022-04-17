from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

import torch.nn as nn
import torch


BERT_MODEL = 'bert-base-uncased'
MAX_SEQ_LENGTH = 64
EPSILON = 1e-5

class Bert(nn.Module):
    def __init__(self, num_labels, is_cuda):
        super(Bert, self).__init__()
        # self._label_list = label_list
        # self._ren = ren
        # self._device = device

        self._tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL, do_lower_case=True)

        self._model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL, num_labels=num_labels)
        
        if is_cuda:
            self._model = self._model.cuda()

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

    def forward(self, x,full_pred=True):
        return self._model(x)

    #     self._weights = None
    #     self._w_decay = None

    #     if norm_fn == 'linear':
    #         self._norm_fn = _linear_normalize
    #     elif norm_fn == 'softmax':
    #         self._norm_fn = _softmax_normalize

    #     if ren:
    #         assert norm_fn == 'linear'

    # def init_weights(self, n_examples, w_init, w_decay):
    #     if self._ren:
    #         raise ValueError(
    #             'no global weighting initialization when \'ren\'=True')

    #     self._weights = torch.tensor(
    #         [w_init] * n_examples, requires_grad=True).to(device=self._device)
    #     self._w_decay = w_decay
