import os
import re
import sys
import json
import time
import torch
import codecs
import pickle
import logging
import argparse
import unicodedata
import numpy as np
from tqdm import tqdm

import prepro_utils
from Constants import UNK_WORD
from transformers import BertTokenizer
from config import preprocess_folder, MODEL_DIR

new_preprocess_folder = './new_cached_features'

logger = logging.getLogger()


# ----------------------------------------------------------------------------------------
# Setting parser
# ----------------------------------------------------------------------------------------
def add_preprocess_opts(parser):
    # source dir
    parser.add_argument('--source_dataset_dir', type=str, default='/home/sunsi/dataset/OpenKP',
                        help="The path to the source data (raw json).")
    parser.add_argument('--output_path', type=str, default=new_preprocess_folder,
                        help="The dir to save preprocess data")
    parser.add_argument("--pretrain_model_path", type=str, default=MODEL_DIR,  # MODEL_DIR = "./DATA/pretrain_model/"
                        help="Path to pre-trained model .")
    parser.add_argument("--cache_folder", type=str, default='bert-base-cased',
                        help="pretrined tokenizer folder path")


# ----------------------------------------------------------------------------------------
# load openkp source datasets
# ----------------------------------------------------------------------------------------
def set_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logfile = logging.FileHandler(args.log_file, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))


def load_json_data(mode):
    '''
    Keys:
        'url', 'VDOM', 'text', 'KeyPhrases'

    opt.source_dataset_dir:
        /DATA2/wangyongbo/ms_openkp/data
    '''
    source_path = os.path.join(opt.source_dataset_dir, 'OpenKP%s.jsonl' % mode)
    data_pairs = []
    with codecs.open(source_path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(tqdm(corpus_file)):
            json_ = json.loads(line)
            data_pairs.append(json_)
    return data_pairs


# ----------------------------------------------------------------------------------------
# tokenize through space , delete null keyphrase
# ----------------------------------------------------------------------------------------
def tokenize_source_data(examples):
    ''' delete null keyphrases & check are there the null keyprhases
    '''
    return_pairs = []
    shuffler = prepro_utils.DEL_ASCII()
    for idx, ex in enumerate(tqdm(examples)):  # examples format, [{}, {}, ...]
        # tokenize :delete double spaces
        src_tokens = [w.strip() for w in ex['text'].split() if shuffler.do(w.strip())]
        # delete punctuation in the tail of keyphrase
        trgs_tokens = []
        for phrase in ex['KeyPhrases']:
            if len(phrase) < 1:
                continue
            clean_phrase = ' '.join(phrase).strip('.').strip(',').strip('?').strip()
            trgs_tokens.append(clean_phrase.split())
            # trgs_tokens = [p for p in ex['KeyPhrases'] if len(p) > 0]

        """
        return_pairs:
            [
                [
                    [w1, w2, w3, ...], 
                    [[kp_w1, kp_w2, ...], ...], 
                    'http://...'
                ], 
                ...
            ]
        """
        return_pairs.append((src_tokens, trgs_tokens, ex['url']))

    return return_pairs


# ----------------------------------------------------------------------------------------
# find keyphrase position in document and merge the same keyphrases
# ----------------------------------------------------------------------------------------
def filter_absent_phrase(examples):
    '''
    delete keyphrases punctuation
    find keyphrase start and end positions in documents

    examples:
        [
            [
                [w1, w2, w3, ...],
                [[kp_w1, kp_w2, ...], ...],
                'http://...'
            ],
            ...
        ]

    :return
        [
            {
                'url': 'http://...',
                'doc_tokens': [w1, w2, w3, ...],
                'keyphrases': [[kp_w1, kp_w2, ...], ...],
                'start_end_pos': [[s1, e1], [s2, e2], ..., [s2_1, e2_1], [s2_2, e2_2], ...]
            },
            ...
        ]
    '''
    data_list = []  # [{}, {}, ...]
    overlap_num = 0

    null_index = []  # all `keyphrase` not found in current text.
    absent_index = []  # part of `keyphrase` not found in current text.
    for idx, ex in enumerate(tqdm(examples)):  # for each sample.

        """
        `lower_tokens`: 
            [w1, w2, w3, ...]
        
        `lower_phrases` (lower), `prepro_phrases` (raw) format :
            [[kp_w1, kp_w2, ...], ...]
        """
        lower_tokens = [t.lower() for t in ex[0]]
        lower_phrases, prepro_phrases = prepro_utils.merge_same(ex[1])
        # lower_phrases : lower cased phrases & prepro_phrases : cased phrases

        """
        `present_phrases`:
            {
                'keyphrases': [answer1 str, answer2 str, ...], 
                'start_end_pos': [ [[s1, e1], [s2, e2], ...], [[s1, e1], [s2, e2], ...], ... ]
            }
        """
        present_phrases = prepro_utils.find_answer(lower_tokens, lower_phrases)
        if present_phrases is None:  # not found corresponding keyphrase in current text.
            null_index.append(idx)
            continue
        if len(present_phrases['keyphrases']) != len(lower_phrases):
            absent_index.append(idx)

        """
        `flatten_postions`, (always)
            [[s1, e1], [s2, e2], ..., [s2_1, e2_1], [s2_2, e2_2], ...]
        """
        # filter overlap label positions
        flatten_postions = [pos for poses in present_phrases['start_end_pos'] for pos in poses]
        sorted_positions = sorted(flatten_postions, key=lambda x: x[0])  # 升序
        filter_positions = prepro_utils.filter_overlap(sorted_positions)
        if len(filter_positions) != len(sorted_positions):
            overlap_num += 1

        data = {}
        data['url'] = ex[-1]  # 'http://...'
        data['doc_tokens'] = ex[0]  # [w1, w2, w3, ...]
        data['keyphrases'] = prepro_phrases  # [[kp_w1, kp_w2, ...], ...]
        data['start_end_pos'] = filter_positions  # [[s1, e1], [s2, e2], ..., [s2_1, e2_1], [s2_2, e2_2], ...]
        data_list.append(data)

    logger.info('%d sample is null, null_index as follow : ' % len(null_index))
    logger.info(null_index)
    logger.info('%d sample is absent, absent_index as follow : ' % len(absent_index))
    logger.info(absent_index)
    logger.info('delete overlap keyphrase : %d , overlap / total = %.2f'
                % (overlap_num, float(overlap_num / len(data_list) * 100)) + '%')

    return data_list


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# convert examples for training bert

def convert_for_bert_tag(examples, tokenizer):
    """
    main preprocess function & verify.

    :param examples:
        [
            {
                'url': 'http://...',
                'doc_tokens': [w1, w2, w3, ...],
                'keyphrases': [[kp_w1, kp_w2, ...], ...],
                'start_end_pos': [[s1, e1], [s2, e2], ..., [s2_1, e2_1], [s2_2, e2_2], ...]
            },
            ...
        ]
    :param tokenizer:
        opt.cache_dir:
            ./DATA/pretrain_model/bert-base-cased
        tokenizer = BertTokenizer.from_pretrained(opt.cache_dir)
    :return:
        [
            {
                'url': 'http://...',
                'ex_id': 0-n Int index,
                'tokens': ['Th##', '##is', 'is', 'a', 'test', ...],
                'label': ['O', 'O', 'B', 'I', 'I', 'E', 'O', 'O', 'O', ...],
                'valid_mask': [1,     0,      1,     1,   1, ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                'tok_to_orig_index': [0,      0,      1,    2,   3,     ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                'orig_tokens': [w1, w2, w3, ...],
                'orig_phrases': [[kp_w1, kp_w2, ...], ...],
                'orig_start_end_pos': [[s1, e1], [s2, e2], ..., [s2_1, e2_1], [s2_2, e2_2], ...]
            },
            ...
        ]
    """

    Features = []  # [{}, {}, ...]
    for (example_index, example) in enumerate(tqdm(examples)):  # for each sample.

        """
        Input sentence: 'This is a test ...'
        
        `valid_mask`, 
            'Th##', '##is', 'is', 'a', 'test', ...
            [1,     0,      1,     1,   1, ...]
                    
        `all_doc_tokens`, (all sub tokens)
            ['Th##', '##is', 'is', 'a', 'test', ...]
        
        `tok_to_orig_index`, (indicate the position in raw whole word)
            'Th##', '##is', 'is', 'a', 'test', ...
            [0,      0,      1,    2,   3,     ...]
        """
        valid_mask = []
        all_doc_tokens = []
        tok_to_orig_index = []
        for (i, token) in enumerate(example['doc_tokens']):
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) < 1:
                sub_tokens = [UNK_WORD]  # UNK_WORD = '[UNK]'
            for num, sub_token in enumerate(sub_tokens):
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                if num == 0:
                    valid_mask.append(1)
                else:
                    valid_mask.append(0)

        """
        `label`: ['O', 'O', 'B', 'I', 'I', 'E', 'O', 'O', 'O', ...]
        
        'O' : non-keyphrase
        'B' : begin word of the keyphrase
        'I' : middle word of the keyphrase
        'E' : end word of the keyphrase
        'U' : single word keyphrase
        """
        label = ['O' for _ in range(len(example['doc_tokens']))]
        for s, e in example['start_end_pos']:
            if s == e:
                label[s] = 'U'

            elif (e - s) == 1:
                label[s] = 'B'
                label[e] = 'E'

            elif (e - s) >= 2:
                label[s] = 'B'
                label[e] = 'E'
                for i in range(s + 1, e):
                    label[i] = 'I'
            else:
                logger.info('ERROR')
                break

        """
        {
            'url': 'http://...',
            'ex_id': 0-n Int index,
            'tokens': ['Th##', '##is', 'is', 'a', 'test', ...],
            'label': ['O', 'O', 'B', 'I', 'I', 'E', 'O', 'O', 'O', ...],
            'valid_mask': [1,     0,      1,     1,   1, ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
            'tok_to_orig_index': [0,      0,      1,    2,   3,     ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
            'orig_tokens': [w1, w2, w3, ...],
            'orig_phrases': [[kp_w1, kp_w2, ...], ...],
            'orig_start_end_pos': [[s1, e1], [s2, e2], ..., [s2_1, e2_1], [s2_2, e2_2], ...]
        }
        """
        InputFeatures = {}
        InputFeatures['url'] = example['url']
        InputFeatures['ex_id'] = example_index  # yes
        InputFeatures['tokens'] = all_doc_tokens  # yes
        InputFeatures['label'] = label
        InputFeatures['valid_mask'] = valid_mask
        InputFeatures['tok_to_orig_index'] = tok_to_orig_index
        InputFeatures['orig_tokens'] = example['doc_tokens']
        InputFeatures['orig_phrases'] = example['keyphrases']  # yes
        InputFeatures['orig_start_end_pos'] = example['start_end_pos']  # yes

        # verify InputFeatures
        if not prepro_utils.verify_ex(InputFeatures):
            continue
        else:
            Features.append(InputFeatures)
    return Features


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# convert Eval examples for Test
def preprocess_for_Eval(examples, tokenizer):
    """
    main preprocess function & verify

    :param examples: [{}, {}, ...], (source data loaded from jsonl)
    :param tokenizer:
    :return:
    """

    Features = []  # [{}, {}, ...]
    shuffler = prepro_utils.DEL_ASCII()
    for (example_index, example) in enumerate(tqdm(examples)):
        doc_tokens = [w.strip() for w in example['text'].split() if shuffler.do(w.strip())]

        valid_mask = []
        all_doc_tokens = []
        tok_to_orig_index = []
        for (i, token) in enumerate(doc_tokens):
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) < 1:
                sub_tokens = [UNK_WORD]
            for num, sub_token in enumerate(sub_tokens):
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                if num == 0:
                    valid_mask.append(1)
                else:
                    valid_mask.append(0)

        """
        {
            'url': 'http://...',
            'ex_id': 0-n Int index,
            'VDOM': xxx,
            'tokens': ['Th##', '##is', 'is', 'a', 'test', ...],
            'valid_mask': [1,     0,      1,     1,   1, ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
            'tok_to_orig_index': [0,      0,      1,    2,   3,     ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
            'orig_tokens': [w1, w2, w3, ...],
        }
        """
        InputFeatures = {}
        InputFeatures['ex_id'] = example_index
        InputFeatures['url'] = example['url']
        InputFeatures['VDOM'] = example['VDOM']
        InputFeatures['orig_tokens'] = doc_tokens
        InputFeatures['tokens'] = all_doc_tokens
        InputFeatures['valid_mask'] = valid_mask
        InputFeatures['tok_to_orig_index'] = tok_to_orig_index

        # verify InputFeatures
        if not prepro_utils.check_tokenize(InputFeatures):
            logger.info('Error Found : ex_id = %d , url = %s' % (example_index, example['url']))
            continue
        else:
            Features.append(InputFeatures)
    return Features


# ----------------------------------------------------------------------------------------
# save data
# ----------------------------------------------------------------------------------------
def save_Dev_truths(examples, filename):
    with open(filename, 'w', encoding='utf-8') as f_pred:
        for ex in tqdm(examples):
            data = {}
            data['url'] = ex['url']
            data['KeyPhrases'] = ex['KeyPhrases']
            f_pred.write("{}\n".format(json.dumps(data)))
        f_pred.close()
    print('Success save Dev_reference to %s' % filename)


def save_preprocess_data(data_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_list, f)
    f.close()


# ----------------------------------------------------------------------------------------
# main function
# ----------------------------------------------------------------------------------------
def main_preocess(input_mode, save_mode):
    """
    opt.cache_dir:
        ./DATA/pretrain_model/bert-base-cased

    DATA:
        OpenKPDev.jsonl
        OpenKPEvalPublic.jsonl
        OpenKPTrain.jsonl

    :param input_mode: `EvalPublic`, `Dev`, `Train`
    :param save_mode: `eval_public`, `valid`, `train`
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained(opt.cache_dir)

    source_data = load_json_data(input_mode)  # [{}, {}, ...]
    logger.info("1/5 success loaded %s data : %d " % (input_mode, len(source_data)))

    if input_mode == 'Dev':
        save_Dev_truths(source_data, os.path.join(opt.output_path, "Dev_candidate.json"))

    # preprocess for training
    if input_mode in ['Train', 'Dev']:
        """
        tokenize_data:
            [
                [
                    [w1, w2, w3, ...], 
                    [[kp_w1, kp_w2, ...], ...], 
                    'http://...'
                ], 
                ...
            ]
        """
        tokenize_data = tokenize_source_data(source_data)
        logger.info('2/5: success tokenize %s data !' % save_mode)

        """
        present_data:
            [
                {
                    'url': 'http://...',
                    'doc_tokens': [w1, w2, w3, ...],
                    'keyphrases': [[kp_w1, kp_w2, ...], ...],
                    'start_end_pos': [[s1, e1], [s2, e2], ..., [s2_1, e2_1], [s2_2, e2_2], ...]
                },
                ...
            ]
        """
        present_data = filter_absent_phrase(tokenize_data)
        logger.info('3/5 : success obtain %s present keyphrase for training bert: %d (filter out : %d)'
                    % (save_mode, len(present_data), (len(tokenize_data) - len(present_data))))

        """
        return_examples: (for trainset or devset)
            [
                {
                    'url': 'http://...',
                    'ex_id': 0-n Int index,
                    'tokens': ['Th##', '##is', 'is', 'a', 'test', ...],
                    'label': ['O', 'O', 'B', 'I', 'I', 'E', 'O', 'O', 'O', ...],
                    'valid_mask': [1,     0,      1,     1,   1, ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                    'tok_to_orig_index': [0,      0,      1,    2,   3,     ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                    'orig_tokens': [w1, w2, w3, ...],
                    'orig_phrases': [[kp_w1, kp_w2, ...], ...],
                    'orig_start_end_pos': [[s1, e1], [s2, e2], ..., [s2_1, e2_1], [s2_2, e2_2], ...]
                },
                ...
            ]
        """
        return_examples = convert_for_bert_tag(present_data, tokenizer)
        logger.info('4/5 : success obtain %s data for training bert .' % save_mode)

    # preprocess for evaluation
    elif input_mode == 'EvalPublic':  # testset
        """
        return_examples: (for testset)
            {
                'url': 'http://...',
                'ex_id': 0-n Int index,
                'VDOM': xxx,
                'tokens': ['Th##', '##is', 'is', 'a', 'test', ...],
                'valid_mask': [1,     0,      1,     1,   1, ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                'tok_to_orig_index': [0,      0,      1,    2,   3,     ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                'orig_tokens': [w1, w2, w3, ...],
            }
        """
        return_examples = preprocess_for_Eval(source_data, tokenizer)
    else:
        logger.info('Error : not this mode : %s' % input_mode)

    """
    opt.output_path: (default)
        './new_cached_features'
    """
    filename = os.path.join(opt.output_path, "openkp.%s.json" % save_mode)
    save_preprocess_data(return_examples, filename)
    logger.info("5/5 : success saved %s data to : %s" % (save_mode, filename))


if __name__ == "__main__":
    """
    Run for data preprocessing,
        python preprocess.py --source_dataset_dir "your own directory" --output_path "your save directory"
    """

    t0 = time.time()
    parser = argparse.ArgumentParser(description='preprocess_openkp.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add options
    opt = ''.split()
    add_preprocess_opts(parser)
    opt = parser.parse_args(opt)

    """
    parser.add_argument('--source_dataset_dir', type=str, default='/home/sunsi/dataset/OpenKP',
                        help="The path to the source data (raw json).")
    parser.add_argument('--output_path', type=str, default=new_preprocess_folder,
                        help="The dir to save preprocess data")
    parser.add_argument("--pretrain_model_path", type=str, default=MODEL_DIR,  # MODEL_DIR = "./DATA/pretrain_model/"
                        help="Path to pre-trained model .")
    parser.add_argument("--cache_folder", type=str, default='bert-base-cased',
                        help="pretrined tokenizer folder path")
    """
    opt.cache_dir = os.path.join(opt.pretrain_model_path, opt.cache_folder)
    opt.log_file = os.path.join(opt.output_path, '%s-logging.txt' % time.strftime("%Y.%m.%d-%H:%M:%S"))

    # folder
    if not os.path.exists(opt.source_dataset_dir):
        logger.info("don't exist the source dataset dir: %s" % opt.source_dataset_dir)

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    # set logging
    set_logger(opt)

    mode_dir = [('EvalPublic', 'eval_public'), ('Dev', 'valid'), ('Train', 'train')]
    for input_mode, save_mode in mode_dir:
        main_preocess(input_mode, save_mode)
