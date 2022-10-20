# coding=utf-8
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import argparse
from transformers import BertTokenizer, XLMTokenizer, XLMRobertaTokenizer
import os
from collections import defaultdict
import csv
import random
import os
import shutil
import json

TOKENIZERS = {
    'bert': BertTokenizer,
    'xlm': XLMTokenizer,
    'xlmr': XLMRobertaTokenizer,
}


def xnli_preprocess(args):
    def _preprocess_file(infile, output_dir, split):
        all_langs = defaultdict(list)
        for i, line in enumerate(open(infile, 'r')):
            if i == 0:
                continue

            items = line.strip().split('\t')
            lang = items[0].strip()
            label = "contradiction" if items[1].strip() == "contradictory" else items[1].strip()
            sent1 = ' '.join(items[6].strip().split(' '))
            sent2 = ' '.join(items[7].strip().split(' '))
            all_langs[lang].append((sent1, sent2, label))
        print(f'# langs={len(all_langs)}')
        for lang, pairs in all_langs.items():
            outfile = os.path.join(output_dir, '{}-{}.tsv'.format(split, lang))
            with open(outfile, 'w') as fout:
                writer = csv.writer(fout, delimiter='\t')
                for (sent1, sent2, label) in pairs:
                    writer.writerow([sent1, sent2, label])
            print(f'finish preprocess {outfile}')

    def _preprocess_train_file(infile, outfile):
        with open(outfile, 'w') as fout:
            writer = csv.writer(fout, delimiter='\t')
            for i, line in enumerate(open(infile, 'r')):
                if i == 0:
                    continue

                items = line.strip().split('\t')
                sent1 = ' '.join(items[0].strip().split(' '))
                sent2 = ' '.join(items[1].strip().split(' '))
                label = "contradiction" if items[2].strip() == "contradictory" else items[2].strip()
                writer.writerow([sent1, sent2, label])
        print(f'finish preprocess {outfile}')

    infile = os.path.join(args.data_dir, 'XNLI-MT-1.0/multinli/multinli.train.en.tsv')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    outfile = os.path.join(args.output_dir, 'train-en.tsv')
    _preprocess_train_file(infile, outfile)

    for split in ['test', 'dev']:
        infile = os.path.join(args.data_dir, 'XNLI-1.0/xnli.{}.tsv'.format(split))
        print(f'reading file {infile}')
        _preprocess_file(infile, args.output_dir, split)


def remove_qa_test_annotations(test_dir):
    assert os.path.exists(test_dir)
    for file_name in os.listdir(test_dir):
        new_data = []
        test_file = os.path.join(test_dir, file_name)
        with open(test_file, 'r') as f:
            data = json.load(f)
            version = data['version']
            for doc in data['data']:
                for par in doc['paragraphs']:
                    context = par['context']
                    for qa in par['qas']:
                        question = qa['question']
                        question_id = qa['id']
                        for answer in qa['answers']:
                            a_start, a_text = answer['answer_start'], answer['text']
                            a_end = a_start + len(a_text)
                            assert context[a_start:a_end] == a_text
                        new_data.append({'paragraphs': [{
                            'context': context,
                            'qas': [{'answers': [{'answer_start': 0, 'text': ''}],
                                     'question': question,
                                     'id': question_id}]}]})
        with open(test_file, 'w') as f:
            json.dump({'data': new_data, 'version': version}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir where any processed files will be written to.")
    parser.add_argument("--task", default="xnli", type=str, required=True,
                        help="The task name")
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                        help="The pre-trained model")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="model type")
    parser.add_argument("--max_len", default=512, type=int,
                        help="the maximum length of sentences")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="whether to do lower case")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="cache directory")
    parser.add_argument("--languages", default="en", type=str,
                        help="process language")
    parser.add_argument("--remove_last_token", action='store_true',
                        help="whether to remove the last token")
    parser.add_argument("--remove_test_label", action='store_true',
                        help="whether to remove test set label")
    args = parser.parse_args()

    # Modification: keep test labels of NER and QA for evaluation
    if args.task == 'xnli':
        xnli_preprocess(args)
