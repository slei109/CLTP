import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import util
import time
from tensorize import XnliDataProcessor
from os.path import join
from datetime import datetime
import sys
from collections import defaultdict
import pickle
import random
from model_xnli import TransformerXnli
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset, ConcatDataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import classification_report
from selection import get_selection_prelim
import torch.nn.functional as F


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class XnliRunner:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info(f'Log file path: {log_path}')

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = XnliDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None, config_name=None):
        num_labels = len(self.data.get_labels())
        model = TransformerXnli(self.config, num_labels)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix, config_name)
        if self.config['freeze_emb']:
            model.freeze_emb()
        return model

    def prepare_inputs(self, batch, with_labels=True):
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'lang_ids': batch[3]
        }
        if with_labels:
            inputs['labels'] = batch[-1]
        return inputs

    def sl_select_dataset(self, model, lang, selected_indices, selected_labels, dev_dataset_by_lang,
                          criterion='max_prob', shortcut_1st_itr=True):
        assert lang != 'en'
        higher_is_better = util.if_higher_is_better(criterion)

        conf = self.config
        dataset = dev_dataset_by_lang[lang]

        # Get dataset for prediction
        prev_indices = {idx for indices in selected_indices[:-1] for idx in indices[lang]}
        remaining_indices = list(set(range(len(dataset))).difference(prev_indices))
        idx_to_i = {idx: i for i, idx in enumerate(remaining_indices)}
        top_k = min(max(int(len(dataset) * conf['sl_top_k_ratio']), 1), len(remaining_indices))
        threshold = conf['sl_selection_threshold']

        # Predict
        if shortcut_1st_itr and len(selected_indices) == 1:  # First itr pred is always same from warm-start
            result_path = join(conf['log_root'], conf['init_config_name'], 'selection', conf['init_suffix'],
                               f'results_{lang}.bin')  # SelectionAnalyzer.evaluate()
            with open(result_path, 'rb') as f:
                results = pickle.load(f)
                metrics, preds, labels, probs, (logits, un) = results  # Order should match initial remaining_indices  logits[2490,3]
            logger.info(f'Shortcut first iteration prediction from {result_path}')
        else:
            metrics, preds, labels, probs, (logits, un) = self.evaluate_simple(model, Subset(dataset, remaining_indices),
                                                                               only_predict=True)
        ### establish partial label
        if conf['partial']:
            raw_pred_onehot = F.one_hot(torch.where(torch.from_numpy(preds) >= 0, torch.from_numpy(preds), 0), logits[0].shape[-1])
            ### top2 label
            _, top2label = torch.topk(torch.from_numpy(logits), 2)
            raw_pred_partial = torch.zeros(raw_pred_onehot.shape)
            raw_pred_partial = raw_pred_partial.scatter(1, top2label, 1)

        if not conf['partial']:
            # Get selection prelim
            scores = get_selection_prelim(preds, probs, logits, un, criterion)
            # Get indices with top scores
            final_idx, _ = self._get_selected_indices(remaining_indices, scores, preds, higher_is_better, top_k, threshold)
        else:
            # Get selection prelim: type_scores
            scores = get_selection_prelim(preds, probs, logits, un, criterion)
            # Get indices with top scores
            full_idx, partial_idx, _ = self._get_selected_indices_partial(remaining_indices, scores, preds, higher_is_better, top_k,
                                                      threshold)

        if not self.config['sl_gold_labels']:  # sl_gold_labels: only for debugging
            if not conf['partial']:
                # Update selected_indices
                selected_indices[-1][lang] = final_idx
                # Update dataset with predicted labels for self-training
                final_idx_i = [idx_to_i[idx] for idx in final_idx]
                dataset.tensors[-1][torch.LongTensor(final_idx)] = torch.as_tensor(preds[final_idx_i], dtype=torch.long)
                selected_labels[-1][lang] = preds[final_idx_i]  # Keep SL state
                return len(final_idx)
            else:
                # Update selected_indices
                dataset_tensor = F.one_hot(dataset.tensors[-1], num_classes=logits[0].shape[-1])
                dataset_list = list(dataset.tensors)
                dataset_list[-1] = dataset_tensor
                # dataset.tensors[-1] = F.one_hot(dataset.tensors[-1], num_classes=logits[0].shape[-1])
                dataset_tuple = tuple(dataset_list)
                dataset.tensors = dataset_tuple
                if conf['full_label']:
                    selected_indices[-1][lang] = full_idx
                    full_idx_i = [idx_to_i[idx] for idx in full_idx]
                    dataset.tensors[-1][torch.LongTensor(full_idx)] = torch.as_tensor(raw_pred_onehot[full_idx_i],
                                                                                      dtype=torch.long)  #### save the prediction label and update the dataset
                    selected_labels[-1][lang] = raw_pred_onehot[full_idx_i]  # onehot label
                selected_indices[-1][lang] = partial_idx
                # Update dataset with predicted tags for self-training
                partial_idx_i = [idx_to_i[idx] for idx in partial_idx]
                dataset.tensors[-1][torch.LongTensor(partial_idx)] = torch.as_tensor(raw_pred_partial[partial_idx_i],
                                                                                     dtype=torch.long)  #### save the prediction label and update the dataset
                selected_labels[-1][lang] = raw_pred_partial[partial_idx_i]  # partial label
                return (len(full_idx) + len(partial_idx))


    def _get_selected_indices(self, indices, scores, preds, higher_is_better, top_k=None, threshold=None):
        idx_to_i = {idx: i for i, idx in enumerate(indices)}
        score_idx = [(score, idx) for score, idx in zip(scores.tolist(), indices)]
        score_idx = sorted(score_idx, reverse=higher_is_better)
        type_scores = defaultdict(list)
        for score, idx in score_idx:
            type_scores[preds[idx_to_i[idx]].item()].append((score, idx))

        def get_selected_indices_per_type(this_type):
            indices = [idx for score, idx in type_scores[this_type]]
            scores = [score for score, idx in type_scores[this_type]]
            num_selection = len(scores)
            if threshold:
                if higher_is_better:
                    num_selection = len(scores) - np.searchsorted(scores[::-1], threshold, side='left')
                else:
                    num_selection = np.searchsorted(scores, threshold, side='right')
            if top_k:
                if top_k > num_selection and threshold:
                    logger.info(f'Throttle type {this_type} selection by threshold: {top_k} to {num_selection}')
                else:
                    num_selection = top_k
            selected_indices = indices[:num_selection]
            return selected_indices

        final_indices = []
        for this_type in type_scores.keys():
            final_indices += get_selected_indices_per_type(this_type)
        return final_indices, type_scores

    def _get_selected_indices_partial(self, indices, scores, preds, higher_is_better, top_k=None, threshold=None):
        idx_to_i = {idx: i for i, idx in enumerate(indices)}
        score_idx = [(score, idx) for score, idx in zip(scores.tolist(), indices)]
        score_idx = sorted(score_idx, reverse=higher_is_better)
        type_scores = defaultdict(list)
        for score, idx in score_idx:
            type_scores[preds[idx_to_i[idx]].item()].append((score, idx))

        def get_selected_indices_per_type(this_type):
            indices = [idx for score, idx in type_scores[this_type]]
            scores = [score for score, idx in type_scores[this_type]]
            num_selection = len(scores)
            if threshold:
                if higher_is_better:
                    num_selection = len(scores) - np.searchsorted(scores[::-1], threshold, side='left')
                else:
                    num_selection = np.searchsorted(scores, threshold, side='right')
            if top_k:
                if top_k > num_selection and threshold:
                    logger.info(f'Throttle type {this_type} selection by threshold: {top_k} to {num_selection}')
                else:
                    num_selection = top_k
            selected_full_indices = indices[:num_selection]
            selected_partial_indices = indices[num_selection*2:]
            return selected_full_indices, selected_partial_indices

        final_full_indices = []
        final_partial_indices = []
        for this_type in type_scores.keys():
            full_indices, partial_indices = get_selected_indices_per_type(this_type)
            final_full_indices += full_indices
            final_partial_indices += partial_indices
        return final_full_indices, final_partial_indices, type_scores

    def _filter_selected_indices_by_gold(self, dataset, preds, final_idx, final_idx_i):
        # Should only be called for debugging
        preds = torch.as_tensor(preds[final_idx_i], dtype=torch.long)
        matches = dataset.tensors[-1][torch.LongTensor(final_idx)] == preds
        final_idx = [idx for idx, mismatch in zip(final_idx, matches.tolist()) if mismatch]
        return final_idx

    def _get_sl_train_dataset(self, en_train_dataset, dev_dataset_by_lang, selected_indices):
        train_dataset = {}
        if not self.config['partial']:
            for lang in util.langs:
                if lang == 'en':
                    num_en = int(len(en_train_dataset) * self.config['sl_en_ratio'])
                    train_dataset[lang] = Subset(en_train_dataset, random.sample(range(len(en_train_dataset)), k=num_en))
                else:
                    if self.config['sl_lang_ratio']: #### self.config['sl_lang_ratio'] is mixture ratio
                        curr_indices = selected_indices[-1][lang]
                        prev_indices = [idx for indices in selected_indices[:-1] for idx in indices[lang]]
                        all_indices = curr_indices + random.sample(prev_indices, k=min(len(prev_indices),
                            int(len(curr_indices) * self.config['sl_lang_ratio'])))
                    else:
                        all_indices = [idx for indices in selected_indices for idx in indices[lang]]
                    train_dataset[lang] = Subset(dev_dataset_by_lang[lang], all_indices)
        else:
            for lang in util.langs:
                if lang == 'en':
                    num_en = int(len(en_train_dataset) * self.config['sl_en_ratio'])
                    dataset_tensor = F.one_hot(en_train_dataset.tensors[-1], num_classes=3)
                    dataset_list = list(en_train_dataset.tensors)
                    dataset_list[-1] = dataset_tensor
                    dataset_tuple = tuple(dataset_list)
                    en_train_dataset.tensors = dataset_tuple
                    train_dataset[lang] = Subset(en_train_dataset, random.sample(range(len(en_train_dataset)), k=num_en))
                else:
                    if self.config['sl_lang_ratio']:
                        curr_indices = selected_indices[-1][lang]
                        prev_indices = [idx for indices in selected_indices[:-1] for idx in indices[lang]]
                        all_indices = curr_indices + random.sample(prev_indices,
                                        k=min(len(prev_indices), int(len(curr_indices) * self.config['sl_lang_ratio'])))  #### self.config['sl_lang_ratio'] is mixture ratio
                    else:
                        all_indices = [idx for indices in selected_indices for idx in indices[lang]]
                    train_dataset[lang] = Subset(dev_dataset_by_lang[lang], all_indices)
        train_dataset = ConcatDataset([ds for lang, ds in train_dataset.items()])
        return train_dataset

    def sl_terminate(self, itr, max_eval_scores):
        return False

    def train_selected(self, model):
        """ Train one round of selected with silver/gold labels """
        conf = self.config
        logger.info(conf)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info(f'Tensorboard summary path: {tb_path}')

        # Set up data
        en_train_dataset = self.data.get_data('train', 'en', only_dataset=True)
        en_dev_dataset = self.data.get_data('dev', 'en', only_dataset=True)
        dev_dataset_by_lang = {lang: self.data.get_data('dev', lang, only_dataset=True) for lang in util.langs if lang != 'en'}
        _, _, selected_indices, selected_labels, _ = self.load_sl_state(conf['state_suffix'], conf['state_config_name'])

        # Identify selected iteration range
        if conf['itr_start'] == -1:  # Use full selected
            itr_start, itr_end = 0, len(selected_indices) - 1  # Inclusive
        else:
            itr_start = conf['itr_start'] or 0
            itr_end = conf['itr_end'] or itr_start

        # Make training set
        train_dataset = {}
        for lang in util.langs:
            if lang == 'en':
                train_dataset[lang] = en_train_dataset  # Always use full en
            else:
                all_indices = [idx for indices in selected_indices[itr_start: itr_end + 1] for idx in indices[lang]]
                if not conf['use_gold']:  # If use silver, set silver labels
                    all_labels = np.concatenate([labels[lang] for labels in selected_labels[itr_start: itr_end + 1]], axis=0)
                    dev_dataset_by_lang[lang].tensors[-1][torch.LongTensor(all_indices)] = torch.as_tensor(all_labels, dtype=torch.long)
                train_dataset[lang] = Subset(dev_dataset_by_lang[lang], all_indices)
        train_dataset = ConcatDataset([ds for lang, ds in train_dataset.items()])

        # Train
        loss_history = []
        max_eval_score = self.train_single(model, train_dataset, en_dev_dataset, loss_history, conf['num_epochs'], tb_writer)
        self.save_model_checkpoint(model, len(loss_history))
        # Eval all langs
        avg_acc, all_acc = self.evaluate_simple_all_langs(model)

    def train_full(self, model, state_suffix=None):
        conf = self.config
        logger.info(conf)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info(f'Tensorboard summary path: {tb_path}')

        # Set up data
        en_train_dataset = self.data.get_data('train', 'en', only_dataset=True)
        en_dev_dataset = self.data.get_data('dev', 'en', only_dataset=True)
        dev_dataset_by_lang = {lang: self.data.get_data('dev', lang, only_dataset=True) for lang in util.langs if lang != 'en'}

        # Initialize SL states
        if state_suffix is None:
            itr, selected_indices, selected_labels = 1, [], []  # itr = 1 if warm-start
            loss_history = []  # Full history of effective loss; length equals total update steps
            max_eval_scores = []
        else:
            loss_history, itr, selected_indices, selected_labels, max_eval_scores = self.load_sl_state(state_suffix)
            if not conf['sl_gold_labels']:
                for lang in util.langs:
                    if lang == 'en':
                        continue
                    all_indices = [idx for indices in selected_indices for idx in indices[lang]]
                    all_labels = np.concatenate([labels[lang] for labels in selected_labels], axis=0)
                    dev_dataset_by_lang[lang].tensors[-1][torch.LongTensor(all_indices)] = torch.as_tensor(all_labels, dtype=torch.long)

        # Start iterative training
        while itr < conf['sl_max_itr']:
            logger.info('=' * 20 + f'SL Iteration {itr}' + '=' * 20)

            train_dataset, epochs = None, None  # For current training iteration
            if itr == 0:
                train_dataset, epochs = en_train_dataset, conf['num_epochs']
            else:
                epochs = conf['sl_num_epochs']
                # Select new training data; update selected_indices in sl_select_dataset()
                num_new_selected = 0
                selected_indices.append({})
                selected_labels.append({})
                for lang in util.langs:
                    if lang != 'en':
                        num_new_selected += self.sl_select_dataset(model, lang, selected_indices, selected_labels,
                                                                   dev_dataset_by_lang, criterion=conf['sl_criterion'])
                logger.info(f'Num newly selected examples: {num_new_selected}')
                # Make new training dataset
                train_dataset = self._get_sl_train_dataset(en_train_dataset, dev_dataset_by_lang, selected_indices)

            # Train
            if not conf['partial']:
                max_eval_score = self.train_single(model, train_dataset, en_dev_dataset, loss_history, epochs, tb_writer)
            else:
                max_eval_score = self.train_partial_single(model, train_dataset, en_dev_dataset, loss_history, epochs, tb_writer)
            max_eval_scores.append(max_eval_score)
            itr += 1

            # Save SL state
            self.save_sl_state(loss_history, itr, selected_indices, selected_labels, max_eval_scores)
            # Eval all langs if needed
            avg_f1, all_f1 = self.evaluate_simple_all_langs(model, itr, tb_writer, print_report=False)
            # Terminate if needed
            if self.sl_terminate(itr, max_eval_scores):
                break

        # Wrap up
        tb_writer.close()
        logger.info('max_eval_scores for each itr: ' + '\t'.join([f'{s: .4f}' for s in max_eval_scores]))
        logger.info('Finished SL')
        return loss_history, max_eval_scores, selected_indices, selected_labels

    def train_single(self, model, train_dataset, eval_dataset, loss_history, epochs=None, tb_writer=None):
        conf = self.config
        epochs = epochs or conf['num_epochs']
        batch_size, grad_accum = conf['batch_size'], conf['gradient_accumulation_steps']

        model.to(self.device)

        # Set up data
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size, drop_last=False)

        # Set up optimizer and scheduler
        total_update_steps = len(train_dataloader) * epochs // grad_accum
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer, total_update_steps)

        # Get model parameters for grad clipping
        trained_params = model.parameters()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(train_dataset))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        max_eval_score = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            for batch in train_dataloader:
                # Forward pass
                model.train()
                inputs = self.prepare_inputs(batch, with_labels=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                loss, _ = model(**inputs)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if conf['max_grad_norm']:
                        torch.nn.utils.clip_grad_norm_(trained_params, conf['max_grad_norm'])
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', scheduler.get_last_lr()[0], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        metrics, _, _, _, _ = self.evaluate_simple(model, eval_dataset, len(loss_history), tb_writer=tb_writer)
                        if metrics['acc'] > max_eval_score:
                            max_eval_score = metrics['acc']
                            # self.save_model_checkpoint(model, len(loss_history))
                        logger.info(f'Max eval score: {max_eval_score:.4f}')
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Eval at the end
        metrics, _, _, _, _ = self.evaluate_simple(model, eval_dataset, len(loss_history), tb_writer=tb_writer)
        if metrics['acc'] > max_eval_score:
            max_eval_score = metrics['acc']
        self.save_model_checkpoint(model, len(loss_history))
        logger.info(f'Max eval score: {max_eval_score:.4f}')
        return max_eval_score

    def train_partial_single(self, model, train_dataset, eval_dataset, loss_history, epochs=None, tb_writer=None):
        conf = self.config
        epochs = epochs or conf['num_epochs']
        batch_size, grad_accum = conf['batch_size'], conf['gradient_accumulation_steps']

        model.to(self.device)

        # Set up data
        traindataset = util.gen_index_dataset(train_dataset)
        train_dataloader = DataLoader(traindataset, sampler=SequentialSampler(traindataset), batch_size=batch_size, drop_last=False)

        # Set up optimizer and scheduler
        total_update_steps = len(train_dataloader) * epochs // grad_accum
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer, total_update_steps)

        # Get model parameters for grad clipping
        trained_params = model.parameters()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(train_dataset))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        max_eval_score = 0
        start_time = time.time()
        model.zero_grad()
        n, c = len(train_dataset), 3
        confidence = torch.ones(n, c) / c
        confidence = confidence.to(self.device)
        for epo in range(epochs):
            for batch_i, batch in enumerate(train_dataloader):
                # Forward pass
                model.train()
                inputs = self.prepare_inputs(batch[0], with_labels=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                index = batch_i * batch_size + torch.arange(0, inputs['labels'].shape[0], step=1)

                loss, _ = model(**inputs, confidence=confidence, index=index)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                loss_during_accum.append(loss.item())

                #### Update the confidence of each parital label
                confidence = util.confidence_update_lw(model, confidence, inputs, index)

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if conf['max_grad_norm']:
                        torch.nn.utils.clip_grad_norm_(trained_params, conf['max_grad_norm'])
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', scheduler.get_last_lr()[0], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        metrics, _, _, _, _ = self.evaluate_simple(model, eval_dataset, len(loss_history), tb_writer=tb_writer)
                        if metrics['acc'] > max_eval_score:
                            max_eval_score = metrics['acc']
                            # self.save_model_checkpoint(model, len(loss_history))
                        logger.info(f'Max eval score: {max_eval_score:.4f}')
                        start_time = time.time()

            avg_f1, all_f1 = self.evaluate_simple_all_langs(model, epo, tb_writer, print_report=False)
        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Eval at the end
        metrics, _, _, _, _ = self.evaluate_simple(model, eval_dataset, len(loss_history), tb_writer=tb_writer)
        if metrics['acc'] > max_eval_score:
            max_eval_score = metrics['acc']
        self.save_model_checkpoint(model, len(loss_history))
        logger.info(f'Max eval score: {max_eval_score:.4f}')
        return max_eval_score

    def predict_w_hidden(self, model, dataset, indices):
        dataset = Subset(dataset, indices)
        logger.info(f'Predicting on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

        model.eval()
        model.to(self.device)
        all_logits, all_labels, all_hiddens = [], [], []
        for batch_i, batch in enumerate(dataloader):
            inputs = self.prepare_inputs(batch, with_labels=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs['output_hidden'] = True

            with torch.no_grad():
                logits, un, hiddens = model(**inputs)
            all_logits.append(logits.detach().cpu())
            all_labels.append(batch[-1])
            all_hiddens.append(hiddens.detach().cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_hiddens = torch.cat(all_hiddens, dim=0).numpy()
        return all_logits, all_labels, all_hiddens

    def evaluate_simple(self, model, dataset, step=0, only_predict=False, tb_writer=None, print_report=False):
        conf = self.config
        logger.info(f'Step {step}: evaluating on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=conf['eval_batch_size'])

        # Get results
        model.eval()
        model.to(self.device)
        all_logits, all_labels, all_un = [], [], []
        for batch_i, batch in enumerate(dataloader):
            inputs = self.prepare_inputs(batch, with_labels=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits, un = model(**inputs)
            all_logits.append(logits.detach().cpu())
            all_labels.append(batch[-1])
            all_un.append(None if un is None else un.detach().cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_un = None if all_un[0] is None else torch.cat(all_un, dim=0).numpy()

        all_probs = TransformerXnli.get_probs(all_logits, evi_un=conf['evi_un'])
        all_preds = all_probs.argmax(axis=-1)

        # Get metrics
        if only_predict:
            return None, all_preds, all_labels, all_probs, (all_logits, all_un)
        metrics = {'acc': util.compute_acc(all_preds, all_labels)}

        if print_report:
            logger.info(f'\n{classification_report(all_labels, all_preds, target_names=self.data.get_labels())}')
        if tb_writer:
            for name, val in metrics.items():
                tb_writer.add_scalar(f'Train_Eval_{name}', val, step)
        return metrics, all_preds, all_labels, all_probs, (all_logits, all_un)

    def evaluate_simple_all_langs(self, model, itr=0, tb_writer=None, print_report=False):
        all_acc = []
        for lang in util.langs:
            dataset = self.data.get_data('test', lang, only_dataset=True)
            metrics, _, _, _, _ = self.evaluate_simple(model, dataset, print_report=print_report)
            all_acc.append(metrics['acc'])

        avg_acc = sum(all_acc) / len(all_acc)
        if tb_writer:
            tb_writer.add_scalar(f'Train_Eval_All_Langs', avg_acc, itr)

        logger.info('Eval all langs (test):')
        logger.info('\n'.join(util.print_all_scores(all_acc, 'acc', with_en=True, latex_scale=100)))
        return avg_acc, all_acc

    def evaluate_simple_selected(self, model, state_suffix, ):
        _, _, selected_indices, _, _ = self.load_sl_state(state_suffix)
        all_acc = []
        for lang in util.langs:
            if lang == 'en':
                continue
            all_indices = [idx for indices in selected_indices for idx in indices[lang]]
            dataset = self.data.get_data('dev', lang, only_dataset=True)  # Selected are from dev
            metrics, _, _, _, _ = self.evaluate_simple(model, Subset(dataset, all_indices))
            all_acc.append(metrics['acc'])

        logger.info('Eval all langs (dev):')
        logger.info('\n'.join(util.print_all_scores(all_acc, 'acc', with_en=False, latex_scale=100)))
        return all_acc

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_param = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(grouped_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps'])
        return optimizer

    def get_scheduler(self, optimizer, total_update_steps):
        if self.config['model_type'] == 'mt5':
            # scheduler = get_constant_schedule(optimizer)
            cooldown_start = int(total_update_steps * 0.7)

            def lr_lambda(current_step: int):
                return 1 if current_step < cooldown_start else 0.3

            return LambdaLR(optimizer, lr_lambda, -1)
        else:
            warmup_steps = int(total_update_steps * self.config['warmup_ratio'])
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_update_steps)
        return scheduler

    def save_model_checkpoint(self, model, step):
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix, config_name=None):
        if config_name is None:
            path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        else:
            path_ckpt = join(self.config['log_root'], config_name, f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)

    def save_sl_state(self, *to_save):
        path = join(self.config['log_dir'], f'sl_state_{self.name_suffix}.bin')
        torch.save(to_save, path)
        logger.info('Saved SL state to %s' % path)

    def load_sl_state(self, suffix, config_name=None):
        if config_name is None:
            path = join(self.config['log_dir'], f'sl_state_{suffix}.bin')
        else:
            path = join(self.config['log_root'], config_name, f'sl_state_{suffix}.bin')
        sl_state = torch.load(path)
        logger.info('Loaded SL state from %s' % path)
        return sl_state


if __name__ == '__main__':
    # Train SL
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = XnliRunner(config_name, gpu_id)
    model = runner.initialize_model(runner.config['init_suffix'], runner.config['init_config_name'])
    runner.train_full(model)

    # Train SL from state
    # config_name, gpu_id, suffix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    # runner = TagRunner(config_name, gpu_id)
    # model = runner.initialize_model(suffix)
    # runner.train_full(model, state_suffix=suffix)

    # Train selected with silver/gold
    # config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    # runner = TagRunner(config_name, gpu_id)
    # model = runner.initialize_model()  # Train new model
    # runner.train_selected(model)
