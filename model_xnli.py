from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.mt5.modeling_mt5 import MT5EncoderModel
from torch import nn
import torch.nn.functional as F
import logging
import torch
import torch.nn.init as init
import util
import math
import numpy as np
from numpy.random import default_rng

logger = logging.getLogger(__name__)


def get_seq_encoder(config):
    if config['model_type'] == 'bert':
        return BertModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlmr':
        return XLMRobertaModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'mt5':
        return MT5EncoderModel.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])


class TransformerXnli(nn.Module):
    rng = default_rng()

    def __init__(self, config, num_labels):
        super().__init__()
        assert not config['evi_un'] or not any([config['lang_un'], config['un']])

        self.config = config
        self.num_labels = num_labels

        self.seq_encoder = get_seq_encoder(config)
        self.seq_config = self.seq_encoder.config
        self.seq_hidden_size = self.seq_config.hidden_size
        if config['dim_reduce']:
            self.seq_hidden_size = self.seq_config.hidden_size // config['dim_reduce']
            self.dim_reduce = self.make_linear(self.seq_config.hidden_size, self.seq_hidden_size, bias=False)

        self.dropout = nn.Dropout(config['dropout_rate'])
        self.output_ffnn = nn.Linear(self.seq_hidden_size, num_labels)

        self.un_ffnn = nn.Linear(self.seq_hidden_size, 1)
        self.mc = 10

        self.emb_lang_un = self.make_emb(len(util.lang_to_id), 1)

    def make_emb(self, dict_size, dim_size, std=None):
        emb = nn.Embedding(dict_size, dim_size)
        if std:
            init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=None):
        linear = nn.Linear(in_features, out_features, bias)
        if std:
            init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def freeze_emb(self):
        for param in self.seq_encoder.embeddings.parameters():
            param.requires_grad = False
        logger.info('Froze encoder embedding')

    @classmethod
    def get_probs(cls, logits, un=None, as_regression=False, mc=20, evi_un=False):
        # Evidential un
        if evi_un:
            evidence = logits
            alpha = evidence + 1  # [batch size, num labels]
            S = alpha.sum(axis=-1, keepdims=True)  # [batch size, 1]
            probs = alpha / S
            return probs
        # No un
        if un is None:
            return util.compute_softmax(logits)
        # Un as softmax temperature
        if as_regression:
            logits *= np.exp(-np.expand_dims(un, axis=-1))
            return util.compute_softmax(logits)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, lang_ids=None, labels=None,
                output_hidden=False, confidence=None, index=None):
        conf, batch_size, seq_len = self.config, input_ids.shape[0], input_ids.shape[1]

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                  'output_attentions': False, 'output_hidden_states': False, 'return_dict': False}
        if conf['model_type'] not in ['mt5']:
            inputs['token_type_ids'] = token_type_ids
        outputs = self.seq_encoder(**inputs)
        cls = outputs[1]  # [batch size, seq hidden]
        if conf['dim_reduce']:
            cls = self.dim_reduce(self.dropout(cls))

        logits = self.output_ffnn(self.dropout(cls))  # [batch size, num labels]
        logits = (F.elu(logits) + 1) if conf['evi_un'] else logits  # Range (0, +INF)

        un = self.un_ffnn(cls).squeeze(-1) if conf['un'] else None  # [batch size]

        # Get loss
        loss = None
        if not conf['partial'] and labels is not None:
            # Evidential un
            if conf['evi_un']:
                evidence = logits
                alpha = evidence + 1  # [batch size, num labels]
                S = alpha.sum(dim=-1, keepdim=True)  # [batch size, 1]
                ###### cross-entropy ######
                onehot_labels = F.one_hot(labels, logits.shape[-1])  # [batch size, num labels]
                loss = torch.sum(onehot_labels * (torch.digamma(S) - torch.digamma(alpha)), dim=-1, keepdim=False)

            if loss is None:  # Normal loss without un
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:  # Loss with un
                loss = loss.mean()
        elif conf['partial'] and labels is not None:
            lw = conf['sigmoid_loss_weight']
            lw0 = conf['lw_of_first_term']

            loss, _, _ = util.lws_loss(logits, labels, confidence, index, lw, lw0)  ### here index is for confidence

        total_output = (logits, un, cls) if output_hidden else (logits, un)
        return total_output if loss is None else (loss, total_output)
