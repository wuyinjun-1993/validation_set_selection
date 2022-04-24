from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertPreTrainedModel, BertModel

import torch.nn as nn
import torch


BERT_MODEL = 'bert-base-uncased'
MAX_SEQ_LENGTH = 64
EPSILON = 1e-5


# class BertForSequenceClassification(BertPreTrainedModel):
#     r"""
#         **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in ``[0, ..., config.num_labels - 1]``.
#             If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
#             If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

#     Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
#         **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#             Classification (or regression if config.num_labels==1) loss.
#         **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
#             Classification (or regression if config.num_labels==1) scores (before SoftMax).
#         **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
#             list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
#             of shape ``(batch_size, sequence_length, hidden_size)``:
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         **attentions**: (`optional`, returned when ``config.output_attentions=True``)
#             list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#     Examples::

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#         labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)
#         loss, logits = outputs[:2]

#     """
#     def __init__(self, config):
#         super(BertForSequenceClassification, self).__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

#         self.init_weights()

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
#                 position_ids=None, head_mask=None):
#         outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
#                             attention_mask=attention_mask, head_mask=head_mask)
#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)


def init_model_with_pretrained_model_weights(model):
    pretrained_model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL, config=model.config)

    pretrained_dict = pretrained_model.state_dict()

    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}


    model.load_state_dict(pretrained_dict)

    return model

class custom_Bert(BertPreTrainedModel):
    def __init__(self, num_labels):
        config = BertConfig.from_pretrained(BERT_MODEL)
        config.num_labels=num_labels

        super(custom_Bert, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()
        # super(Bert, self).__init__()
        # self._label_list = label_list
        # self._ren = ren
        # self._device = device

        self._tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL, do_lower_case=True)
        
        

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}
        self.init_model_with_pretrained_model_weights()
        # print()

    # def forward(self, x,full_pred=True):
    #     input_ids, input_mask, segment_ids = x

    #     return self.bert(input_ids, segment_ids, input_mask)[0]

    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
    def init_model_with_pretrained_model_weights(self):
        pretrained_model = BertForSequenceClassification.from_pretrained(
                BERT_MODEL, config=self.config)

        self.load_state_dict(pretrained_model.state_dict())
        

    def forward(self, x, position_ids=None, head_mask=None):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # if labels is not None:
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs

        return outputs[0]  # (loss), logits, (hidden_states), (attentions)

    def feature_forward(self, x, all_layer=False):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=None)
        pooled_output = self.dropout(outputs[1])

        return pooled_output

    def obtain_gradient_last_full_layer(self, sample_representation_last_layer, target, criterion, is_cuda = False):
        output = self.classifier(sample_representation_last_layer)
        loss = criterion(output, target)
        onehot_target = torch.zeros(output.shape[1])
        onehot_target[target] = 1
        if is_cuda:
            onehot_target = onehot_target.cuda()

        total_loss = loss + torch.sum(onehot_target.view(-1)*output.view(-1))
        sample_representation_last_layer_grad = torch.autograd.grad(total_loss, sample_representation_last_layer)[0]
        return sample_representation_last_layer_grad
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
