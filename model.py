import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
# from apex import amp

from utils import override_args
from networks import BertForSeqTagging
from transformers import BertConfig, AdamW, WarmupLinearSchedule

logger = logging.getLogger()


class KeyphraseSpanExtraction(object):

    def __init__(self, args, state_dict=None):

        self.args = args
        self.updates = 0

        """
        `args.cache_dir`, bert-base-cased,
        `args.num_labels`, 5
        """
        model_config = BertConfig.from_pretrained(args.cache_dir, num_labels=args.num_labels)
        self.network = BertForSeqTagging.from_pretrained(args.cache_dir, config=model_config)

        if state_dict is not None:
            self.network.load_state_dict(state_dict)
            logger.info('loaded pretrain model state_dict')

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def init_optimizer(self, num_total_steps):
        """
        `args.warmup_proportion` : Linear warmup over warmup_ratio warm_step / t_total,
        `named_parameters()` : yielding both the name of the parameter as well as the parameter itself
        """
        num_warmup_steps = int(self.args.warmup_proportion * num_total_steps)
        logger.info('warmup steps : %d' % num_warmup_steps)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']  # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        param_optimizer = list(self.network.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},  # weight_decay default=0.01
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # `learning_rate` default=5e-5
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=False)
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def update(self, step, batch):  # for one batch
        # Train mode
        self.network.train()  # Sets the module in training mode.

        """
        TODO: Why the first-5 in batch???
        
        The reason are :
            batch = [input_ids, input_mask, valid_ids, active_mask, labels, ids]  (for `train`)
            
            batch = [input_ids, input_mask, valid_ids, active_mask, valid_lens, ids]  (for `test`)
        """
        # Transfer to GPU
        inputs = [b.to(self.args.device) for b in batch[:5]]

        """
        `self.network(*inputs)` :
            列表前面加星号作用是将列表解开成两个独立的参数，传入函数，
            字典前面加两个星号，是将字典解开成独立的元素作为形参。
        
        Examples:
            def add(a, b):
                return a+b
             
            data = [4,3]
            print add(*data)
            #equals to print add(4, 3)
            
            data = {'a' : 4, 'b' : 3}
            print add(**data)
            #equals to print add(4, 3)
        """
        # run !
        loss = self.network(*inputs)

        if self.args.n_gpu > 1:
            # mean() to average on multi-gpu parallel (not distributed) training
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:  # TODO: amp?
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad()
            self.updates += 1

        return loss.item()

    def predict(self, batch):
        self.network.eval()

        inputs = [b.to(self.args.device) for b in batch[:5]]

        with torch.no_grad():
            loss = self.network(*inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()

        return loss.item()

    def test(self, batch):
        self.network.eval()

        inputs = [b.to(self.args.device) for b in batch[:4]]

        with torch.no_grad():
            logits = self.network(*inputs)
            logits = F.softmax(logits, dim=-1)
        logits = logits.data.cpu()

        return self.decode(logits, batch[4])

    @staticmethod
    def decode(logits, lengths):
        assert logits.size(0) == sum(lengths)

        logits = logits.tolist()
        logit_list = []

        sum_len = 0
        for l in lengths:
            logit_list.append(logits[sum_len:sum_len + l])
            sum_len += l

        return logit_list

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def save_checkpoint(self, filename, epoch):
        network = self.network.module if hasattr(self.network, 'module') else self.network
        params = {
            'args': self.args,
            'epoch': epoch,
            'state_dict': network.state_dict(),
        }
        try:
            torch.save(params, filename)
            logger.info('success save epoch_%d checkpoints !' % epoch)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load_checkpoint(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(filename, map_location=lambda storage, loc: storage)

        args = saved_params['args']
        epoch = saved_params['epoch']
        state_dict = saved_params['state_dict']

        if new_args:
            args = override_args(args, new_args)

        model = KeyphraseSpanExtraction(args, state_dict)
        logger.info('success loaded epoch_%d checkpoints !' % epoch)
        return model

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def zero_grad(self):
        self.optimizer.zero_grad()  # self.network.zero_grad()

    def set_device(self):
        self.network.to(self.args.device)

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)

    def distribute(self):
        self.distributed = True
        self.network = torch.nn.parallel.DistributedDataParallel(self.network,
                                                                 device_ids=[self.args.local_rank],
                                                                 output_device=self.args.local_rank,
                                                                 find_unused_parameters=True)
