import torch
import logging
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from utils import override_args
from transformers import BertForTokenClassification
from transformers import AdamW, WarmupLinearSchedule

logger = logging.getLogger()


class BertForSeqTagging(BertForTokenClassification):
    """
    nn.Module
        - `PreTrainedModel` : from_pretrained
            - `BertPreTrainedModel` : An abstract class to handle weights initialization and a simple interface for
                                      dowloading and loading pretrained models.
                - `BertForTokenClassification`
                    - `BertForSeqTagging`
    """

    def forward(self, input_ids, attention_mask, valid_ids, active_mask, labels=None):
        """
        `input_ids`, tensor: shape=batch样本个数 × 最大样本长度（即样本中词个数最大值）,
        `input_mask`, tensor: shape=batch样本个数 × 最大样本长度, 即把样本长度个位置用1填充，其余为0,
        `valid_ids`, tensor: shape=batch样本个数 × 最大验证样本长度（即验证样本中词个数最大值）,
        `active_mask`, tensor: shape=batch样本个数 × 最大样本长度, 即把样本长度个位置用1填充，其余为0,
        `labels`, tensor: shape=batch样本个数 × 最大样本长度（即样本中词个数最大值）,
        `ids`, list: [idx1, idx2, idx3, ...], 当前样本在examples中的index

        for `train` :
            input_ids, input_mask, valid_ids, active_mask, labels

        for `test` :
            input_ids, input_mask, valid_ids, active_mask, valid_lens

        :param input_ids: input_ids
        :param attention_mask: input_mask
        :param valid_ids: valid_ids
        :param active_mask: active_mask
        :param labels: labels / valid_lens
        :return:
        """

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        sequence_output = outputs[0]

        batch_size, max_len, feature_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feature_dim,
                                   dtype=torch.float32, device='cuda')
        # get valid outputs : first tokens
        for i in range(batch_size):
            k = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    k += 1
                    valid_output[i][k] = sequence_output[i][j]

        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        """
        >>> a
        tensor([[1, 2, 3],
                [2, 1, 1]])

        >>> bb
        tensor([[ 2.1100,  3.1200,  2.4300],  # 0
                [ 1.4400,  5.2200, 11.3200],  # 1
                [31.4500,  2.1100,  3.1200],  # 2
                [ 2.4300,  1.4400,  5.2200],  # 3
                [11.3200, 31.4500,  2.1100],  # 4
                [ 3.1200,  2.4300,  1.4400],  # 5
                [ 5.2200, 11.3200, 31.4500]]) # 6

        >>> a.view(-1)
        tensor([1, 2, 3, 2, 1, 1])

        >>> bb.shape
        torch.Size([7, 3])
        >>> a.shape
        torch.Size([2, 3])
        >>> a.view(-1).shape
        torch.Size([6])

        >>> bb[a.view(-1)]
        tensor([[ 1.4400,  5.2200, 11.3200],
                [31.4500,  2.1100,  3.1200],
                [ 2.4300,  1.4400,  5.2200],
                [31.4500,  2.1100,  3.1200],
                [ 1.4400,  5.2200, 11.3200],
                [ 1.4400,  5.2200, 11.3200]])
        
        i.e., select line from `bb`.
        
        >>> a.view(-1)
        tensor([1, 2, 3, 2, 1, 1])
        >>> a.view(-1) == 1
        tensor([1, 0, 0, 0, 1, 1], dtype=torch.uint8)
        
        i.e., if element in `a.view(-1)` is `1`, set 1, otherwise, set 0
        
        >>> c = torch.tensor([1,1,0,0,0,0,0], dtype=torch.uint8)
        >>> bb[c]
        tensor([[ 2.1100,  3.1200,  2.4300],
                [ 1.4400,  5.2200, 11.3200]])
        
        i.e., remain the first 2 lines in `bb`, discard the rest.
        """
        active_loss = active_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]

        if labels is not None:  # for training
            loss_fct = CrossEntropyLoss()
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:  # for dev and test
            return active_logits
