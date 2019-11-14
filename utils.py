import os
import time
import json
import torch
import string
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
logger = logging.getLogger(__name__)
from Constants import BOS_WORD, EOS_WORD, DIGIT_WORD


# -------------------------------------------------------------------------------------------
# load datasets or preprocess features
# -------------------------------------------------------------------------------------------
def read_openkp_examples(args, tokenizer):
    """
    load preprocess cached_features files.

    `train`, train set,
    `valid`, dev set,
    `eval_public`, test set

    args.preprocess_folder :
        "./DATA/cached_features"
    :return:
        for `train`:
        {
            'train': [{'t': 111, 'd': 222}, {'t1': 333, 'd1': 555}, ...],
            'valid': [{'t': 111, 'd': 222}, {'t1': 333, 'd1': 555}, ...]
        }

        for `generate`:
        {
            'eval_public': [{'t': 111, 'd': 222}, {'t1': 333, 'd1': 555}, ...],
            'valid': [{'t': 111, 'd': 222}, {'t1': 333, 'd1': 555}, ...]
        }
    """

    if not os.listdir(args.preprocess_folder):
        logger.info('Error : not found %s' % args.preprocess_folder)  # The data has been preprocessed

    if args.run_mode == 'train':  # train model
        mode_dict = {'train': [], 'valid': []}
    elif args.run_mode == 'generate':  # generate prediction
        mode_dict = {'eval_public': [], 'valid': []}
    else:
        raise Exception("Invalid run mode %s!" % args.run_mode)

    for mode in mode_dict:
        filename = os.path.join(args.preprocess_folder, "openkp.%s.json" % mode)
        logger.info("start loading openkp %s data ..." % mode)
        with open(filename, "r", encoding="utf-8") as f:
            mode_dict[mode] = json.load(f)  # [{'t': 111, 'd': 222}, {'t1': 333, 'd1': 555}, ...]
        f.close()
        logger.info("success loaded openkp %s data : %d " % (mode, len(mode_dict[mode])))
    return mode_dict


# -------------------------------------------------------------------------------------------
# build dataset and dataloader
# -------------------------------------------------------------------------------------------        
class build_openkp_dataset(Dataset):
    ''' build datasets for train & eval '''

    def __init__(self, args, examples, tokenizer, converter, shuffle=False):
        self.run_mode = args.run_mode
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src_len = args.max_src_len  # default=510
        self.converter = converter
        if shuffle:
            random.seed(args.seed)  # default=42
            random.shuffle(self.examples)  # [{}, {}, ...], samples list.

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return convert_examples_to_features(index, self.examples[index], self.tokenizer,
                                            self.converter, self.max_src_len, self.run_mode)


def convert_examples_to_features(index, ex, tokenizer, converter, max_src_len, run_mode):
    """
    convert each batch data to tensor ; add [CLS] [SEP] tokens ; cut over 512 tokens.

    :param index: 0-n, all samples index.
    :param ex: (one of samples)
        for `train` or `dev`:
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
        for `test`:
                {
                    'url': 'http://...',
                    'ex_id': 0-n Int index,
                    'VDOM': xxx,
                    'tokens': ['Th##', '##is', 'is', 'a', 'test', ...],
                    'valid_mask': [1,     0,      1,     1,   1, ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                    'tok_to_orig_index': [0,      0,      1,    2,   3,     ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                    'orig_tokens': [w1, w2, w3, ...],
                }
    :param tokenizer:
    :param converter:
    :param max_src_len: default=510
    :param run_mode:

    BOS_WORD = '[CLS]'
    EOS_WORD = '[SEP]'

    for `train`: (for one sample.)
        :return: index, src_tensor,                 valid_mask,                                     label_tensor
                    0   token_id_list               the first sub_token is 1 in one whole word      each sub_token's label (tag -> id)
                        tensor[23, 62, 90, ...]     tensor[0, 1, 0, 1, ..., 0]                      tensor[2, 1, 3, ...]
                        ['Th##', '##is', ...]       [[CLS], 'Th##', '##is', ..., [SEP]]             ['O', 'B', 'I', 'E', 'O', ...]

    for `generate`:
        :return: index, src_tensor,                 valid_mask,                                     valid_orig_doc_len
                    0   token_id_list               the first sub_token is 1 in one whole word      len
                        tensor[23, 62, 90, ...]     tensor[0, 1, 0, 1, ..., 0]                      int
                        ['Th##', '##is', ...]       [[CLS], 'Th##', '##is', ..., [SEP]]             int
    """

    src_tokens = [BOS_WORD] + ex['tokens'][:max_src_len] + [EOS_WORD]  # max_src_len = 510
    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))  # [23, 62, 90, ...]

    valid_ids = [0] + ex['valid_mask'][:max_src_len] + [0]
    valid_mask = torch.LongTensor(valid_ids)  # [0, 1, 0, 1, ..., 0]

    if run_mode == 'train':
        if len(ex['tokens']) < max_src_len:
            orig_max_src_len = max_src_len
        else:
            orig_max_src_len = ex['tok_to_orig_index'][max_src_len - 1] + 1
        label_tensor = torch.LongTensor(converter.convert_tag2idx(ex['label'][:orig_max_src_len]))  # [2, 1, 3, ...]
        return index, src_tensor, valid_mask, label_tensor

    elif run_mode == 'generate':
        valid_orig_doc_len = sum(valid_ids)
        return index, src_tensor, valid_mask, valid_orig_doc_len

    else:
        logger.info('not the mode : %s' % run_mode)


def batchify_features_for_train_eval(batch):
    """
    train dataloader & eval dataloader.

    index, src_tensor, valid_mask, label_tensor        (for `train` or `dev`)
    index, src_tensor, valid_mask, valid_orig_doc_len  (for `test`)

    :param batch: where is `batch` comes from? TODO:
    :return:
        `input_ids`, tensor: shape=batch样本个数 × 最大样本长度（即样本中词个数最大值）,
        `input_mask`, tensor: shape=batch样本个数 × 最大样本长度, 即把样本长度个位置用1填充，其余为0,
        `valid_ids`, tensor: shape=batch样本个数 × 最大验证样本长度（即验证样本中词个数最大值）,
        `active_mask`, tensor: shape=batch样本个数 × 最大样本长度, 即把样本长度个位置用1填充，其余为0,
        `labels`, tensor: shape=batch样本个数 × 最大样本长度（即样本中词个数最大值）,
        `ids`, list: [idx1, idx2, idx3, ...], 当前样本在examples中的index
    """

    ids = [ex[0] for ex in batch]  # index
    docs = [ex[1] for ex in batch]  # src_tensor, tensor (word ids list)
    valid_mask = [ex[2] for ex in batch]  # valid_mask, tensor
    label_list = [ex[3] for ex in batch]  # label_list

    # ---------------------------------------------------------------
    # src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()  # shape=(batch_size, doc_max_length)
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
    # segment_ids = torch.LongTensor(len(docs), doc_max_length).zero_()

    for i, d in enumerate(docs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)

    # ---------------------------------------------------------------
    # valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, :v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # label tensor
    labels = torch.LongTensor(len(label_list), doc_max_length).zero_()
    active_mask = torch.LongTensor(len(label_list), doc_max_length).zero_()
    for i, t in enumerate(label_list):
        labels[i, :t.size(0)].copy_(t)
        active_mask[i, :t.size(0)].fill_(1)

    assert input_ids.size() == valid_ids.size() == labels.size()

    return input_ids, input_mask, valid_ids, active_mask, labels, ids


def batchify_features_for_test(batch):
    ''' test dataloader for Dev & Public_Valid.'''

    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    valid_lens = [ex[3] for ex in batch]

    # ---------------------------------------------------------------
    # src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()

    for i, d in enumerate(docs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)

    # ---------------------------------------------------------------
    # valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, :v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # valid length tensor
    active_mask = torch.LongTensor(len(valid_lens), doc_max_length).zero_()
    for i, l in enumerate(valid_lens):
        active_mask[i, :l].fill_(1)

    assert input_ids.size() == valid_ids.size() == active_mask.size()
    return input_ids, input_mask, valid_ids, active_mask, valid_lens, ids


# -------------------------------------------------------------------------------------------
# other utils fucntions
# -------------------------------------------------------------------------------------------     
def override_args(old_args, new_args):
    ''' cover old args to new args, log which args has been changed.'''

    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            logger.info('Overriding saved %s: %s --> %s' %
                        (k, old_args[k], new_args[k]))
            old_args[k] = new_args[k]
    return argparse.Namespace(**old_args)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
