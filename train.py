import os
import time
import tqdm
import json
import torch
import random
import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm

import utils
import config

logger = logging.getLogger()
torch.backends.cudnn.benchmark = True

from Constants import Idx2Tag
from prepro_utils import IdxTag_Converter
from model import KeyphraseSpanExtraction
from transformers import BertTokenizer
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, data_loader, model, stats, writer):
    """
    :param data_loader: train_data_loader
    :param model: model = KeyphraseSpanExtraction(args)
    :param stats: stats = {'timer': utils.Timer(), 'epoch': 0, 'min_eval_loss': float("inf")}
    :param writer: tb_writer, for tf
    :return:
    """
    logger.info("start training %d epoch ..." % stats['epoch'])

    train_loss = utils.AverageMeter()  # Computes and stores the average and current value.
    epoch_time = utils.Timer()

    epoch_loss = 0
    epoch_step = 0

    for step, batch in enumerate(tqdm(data_loader)):
        """
        batch = [input_ids, input_mask, valid_ids, active_mask, labels, ids]
        """
        try:
            loss = model.update(step, batch)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        train_loss.update(loss)

        epoch_loss += loss
        epoch_step += 1

        if step % args.display_iter == 0:

            if args.local_rank in [-1, 0] and args.use_viso:
                writer.add_scalar('train/loss', train_loss.avg, model.updates)
                writer.add_scalar('train/lr', model.scheduler.get_lr()[0], model.updates)

            logging.info('train: Epoch = %d | iter = %d/%d | ' %
                         (stats['epoch'], step, len(train_data_loader)) +
                         'loss = %.2f | %d updates | elapsed time = %.2f (s) \n' %
                         (train_loss.avg, model.updates, stats['timer'].time()))
            train_loss.reset()

    logging.info('Epoch Mean Loss = %.4f ( Epoch = %d ) | Time for epoch = %.2f (s) \n' %
                 ((epoch_loss / epoch_step), stats['epoch'], epoch_time.time()))


def evaluate(args, data_loader, model, stats, writer):
    logger.info("start evaluate valid ( %d epoch ) ..." % stats['epoch'])

    epoch_time = utils.Timer()

    epoch_loss = 0
    epoch_step = 0

    for step, batch in enumerate(tqdm(data_loader)):
        try:
            loss = model.predict(batch)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        epoch_loss += loss
        epoch_step += 1

    eval_loss = float((epoch_loss / epoch_step))

    if args.local_rank in [-1, 0] and args.use_viso:
        writer.add_scalar('valid/loss', eval_loss, stats['epoch'])

    logging.info('Valid Evaluation | Epoch Mean Loss = %.4f ( Epoch = %d ) | Time for epoch = %.2f (s) \n' %
                 (eval_loss, stats['epoch'], epoch_time.time()))

    return eval_loss


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    python train.py --run_mode train \
    --local_rank -1 \
    --max_train_epochs 5 \
    --save_checkpoint \
    --log_dir ./Log \
    --output_dir ./output \
    """

    # setting args
    parser = argparse.ArgumentParser('BERT-Seq-Tagging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_default_args(parser)

    args = parser.parse_args()

    # initialize args & config
    config.init_args_config(args)
    args.num_labels = len(Idx2Tag)  # Idx2Tag = ['O', 'B', 'I', 'E', 'U']

    # -------------------------------------------------------------------------------------------
    # 1.远程debug
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # -------------------------------------------------------------------------------------------
    # 2.设置GPU环境
    # Setup CUDA, GPU & distributed training
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # -------------------------------------------------------------------------------------------
    # 3.随机种子
    set_seed(args)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # -------------------------------------------------------------------------------------------
    # 4.构建分词对象tokenizer，构建idx和tag转换字典
    # init tokenizer & Converter 
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir)  # bert-base-cased
    converter = IdxTag_Converter(Idx2Tag)  # Idx2Tag = ['O', 'B', 'I', 'E', 'U']

    # -------------------------------------------------------------------------------------------
    # 5.加载数据
    # build dataloaders
    logger.info("start loading openkp datasets ...")
    """
    `args.run_mode` in ['train', 'generate'] :
        dataset_dict for `train`:
        {
            'train': [{}, {}, ...],
            'valid': [{}, {}, ...]
        }

        dataset_dict for `generate`:
        {
            'eval_public': [{}, {}, ...],
            'valid': [{}, {}, ...]
        }
        
        --------------------
            for `train` and `valid` :
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
        
            for `eval_public` :
            [
                {
                    'url': 'http://...',
                    'ex_id': 0-n Int index,
                    'VDOM': xxx,
                    'tokens': ['Th##', '##is', 'is', 'a', 'test', ...],
                    'valid_mask': [1,     0,      1,     1,   1, ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                    'tok_to_orig_index': [0,      0,      1,    2,   3,     ...],  # 'Th##', '##is', 'is', 'a', 'test', ...
                    'orig_tokens': [w1, w2, w3, ...],
                },
                ...
            ]
    """
    dataset_dict = utils.read_openkp_examples(args, tokenizer)

    # 5.1构建训练数据
    # train dataloader
    """
    `args.per_gpu_train_batch_size`: Batch size per GPU/CPU for training.
    
    `train_dataset` : 
        index, src_tensor, valid_mask, label_tensor        (for `train` or `dev`)
        index, src_tensor, valid_mask, valid_orig_doc_len  (for `test`)
    """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 转为id，转为tensor
    train_dataset = utils.build_openkp_dataset(args, dataset_dict['train'], tokenizer, converter)

    """
    set local_rank=0 for distributed training on multiple gpus.
    """
    train_sampler = torch.utils.data.sampler.RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    """
    `torch.utils.data.DataLoader`:
        Data loader. Combines a dataset and a sampler, and provides
        single- or multi-process iterators over the dataset.
    
    `args.data_workers`, 
        Number of subprocesses for data loading, default=2
    """
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify_features_for_train_eval,  # merges a list of samples to form a mini-batch
        pin_memory=args.cuda,  # bool
    )

    # 5.2构建验证数据
    # valid dataset
    args.valid_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # 转为id，转为tensor
    valid_dataset = utils.build_openkp_dataset(args, dataset_dict['valid'], tokenizer,
                                               converter)  # don't use DistributedSampler

    valid_sampler = torch.utils.data.sampler.SequentialSampler(valid_dataset)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        sampler=valid_sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify_features_for_train_eval,
        shuffle=False,
        pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------------------------
    # 6.设置总训练步数
    # Set training total steps
    if args.max_train_steps > 0:
        t_total = args.max_train_steps
        args.max_train_epochs = args.max_train_steps // (len(train_data_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.max_train_epochs

    # -------------------------------------------------------------------------------------------
    # Preprare Model & Optimizer
    # -------------------------------------------------------------------------------------------
    # 7.初始化模型和优化器
    logger.info(" ************************** Initialize Model & Optimizer ************************** ")

    """
    `args.checkpoint_file`, loaded checkpoint model continue training.
    `args.load_checkpoint`, default=False
    """
    if args.load_checkpoint and os.path.isfile(args.checkpoint_file):
        model = KeyphraseSpanExtraction.load_checkpoint(args.checkpoint_file, args)
    else:
        logger.info('Training model from scratch...')
        model = KeyphraseSpanExtraction(args)
    model.init_optimizer(num_total_steps=t_total)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.set_device()
    if args.n_gpu > 1:
        model.parallelize()

    if args.local_rank != -1:
        model.distribute()

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.local_rank in [-1, 0] and args.use_viso:
        tb_writer = SummaryWriter(args.viso_folder)
    else:
        tb_writer = None

    logger.info("Training/evaluation parameters %s", args)
    logger.info(" ************************** Running training ************************** ")
    logger.info("  Num Train examples = %d", len(train_dataset))
    logger.info("  Num Train Epochs = %d", args.max_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))

    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info(" *********************************************************************** ")

    # -------------------------------------------------------------------------------------------
    # start training
    # -------------------------------------------------------------------------------------------
    # 8.开始训练
    model.zero_grad()
    stats = {'timer': utils.Timer(), 'epoch': 0, 'min_eval_loss': float("inf")}

    for epoch in range(1, (args.max_train_epochs + 1)):
        stats['epoch'] = epoch

        # train 
        train(args, train_data_loader, model, stats, tb_writer)

        # eval & test
        eval_loss = evaluate(args, valid_data_loader, model, stats, tb_writer)
        if eval_loss < stats['min_eval_loss']:
            stats['min_eval_loss'] = eval_loss
            logger.info(" *********************************************************************** ")
            logger.info('Update Min Eval_Loss = %.6f (epoch = %d)' % (stats['min_eval_loss'], stats['epoch']))

        # Checkpoint
        if args.save_checkpoint:
            model.save_checkpoint(os.path.join(args.output_folder, 'epoch_{}.checkpoint'.format(epoch)), stats['epoch'])
