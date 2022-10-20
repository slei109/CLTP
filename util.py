from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import logging
import torch
import random
from transformers import BertTokenizer, XLMRobertaTokenizer, T5Tokenizer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

langs = 'en ar bg de el es fr hi ru sw th tr ur vi zh'.split()
lang_to_id = {lang: i for i, lang in enumerate(langs)}  # Fixed


pad_token_label_id = CrossEntropyLoss().ignore_index
assert pad_token_label_id < 0


def flatten(l):
    return [item for sublist in l for item in sublist]


def if_higher_is_better(criterion):
    assert criterion in ['max_prob', 'entropy', 'var', 'vacuity', 'dissonance', 'custom']
    return True if criterion == 'max_prob' else False


def compute_acc(preds, golds):
    return (preds == golds).mean().item()


def compute_softmax(scores):
    """ axis: -1 """
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return probs


def compute_entropy(probs):
    # probs: [num examples, num classes, ...]
    # Entropy per class per example normalized by num of classes
    num_cls = probs.shape[1]
    return -probs * (np.log(probs) / np.log(num_cls))


def initialize_config(config_name, create_dir=True):
    logger.info("Experiment: {}".format(config_name))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[config_name]

    config['log_dir'] = join(config["log_root"], config_name)
    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    if create_dir:
        makedirs(config['log_dir'], exist_ok=True)
        makedirs(config['tb_dir'], exist_ok=True)

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def print_all_scores(all_scores, score_name, with_en, latex_scale=None):
    print_str = []

    avg_score = sum(all_scores) / len(all_scores)
    print_str.append(f'Avg {score_name}: {avg_score:.4f}')

    print_str.append('LaTex table:')
    all_scores = all_scores if with_en else ([0] + all_scores)  # If without en, set 0
    all_scores = [f'{s * latex_scale :.1f}' if latex_scale else f'{s:.2f}' for s in all_scores]
    print_str.append(' & '.join(all_scores))

    return print_str


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)


def get_bert_tokenizer(config):
    # Avoid using fast tokenization
    if config['model_type'] == 'mt5':
        return T5Tokenizer.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'bert':
        return BertTokenizer.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlmr':
        return XLMRobertaTokenizer.from_pretrained(config['pretrained'])
    else:
        raise ValueError('Unknown model type')


def lws_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0):
    device = outputs.device
    #### update the english dataset, change label to onehot
    counter_onezero = 1 - partialY
    counter_onezero = counter_onezero.to(device)
    sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss1 = sig_loss1.to(device)
    sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
    sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
        1 + torch.exp(-outputs[outputs > 0]))
    l1 = confidence[index, :] * partialY * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss2 = sig_loss2.to(device)
    sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
    sig_loss2[outputs < 0] = torch.exp(
        outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
    l2 = confidence[index, :] * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, lw_weight0 * average_loss1, lw_weight * average_loss2

def confidence_update_lw(model, confidence, inputs, index=None):
    with torch.no_grad():
        device = inputs['labels'].device
        loss, total_ouputs = model(**inputs, confidence=confidence, index=index)
        batch_outputs = total_ouputs[0]
        sm_outputs = F.softmax(batch_outputs, dim=1)

        labels = inputs['labels']
        counter_onezero = 1 - labels
        counter_onezero = counter_onezero.to(device)

        new_weight1 = sm_outputs * labels
        new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight2 = sm_outputs * counter_onezero
        new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight = new_weight1 + new_weight2

        confidence[index,:] = new_weight
        return confidence

class gen_index_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]

        return record, index