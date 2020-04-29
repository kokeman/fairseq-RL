# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import numpy as np

# from transformers import *
from transformers import (BertForSequenceClassification, BertTokenizer,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          AlbertForSequenceClassification, AlbertTokenizer)
from transformers import InputExample, InputFeatures
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)



class Scorer(object):
    def __init__(self, args):
        self.args = args
        # self.model = BertForSequenceClassification.from_pretrained(args.bert_dir)
        self.model_g = BertForSequenceClassification.from_pretrained(args.g_dir)
        self.model_f = BertForSequenceClassification.from_pretrained(args.f_dir)
        self.model_m = BertForSequenceClassification.from_pretrained(args.m_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.g_dir)

    def add(self, src, pred):
        # make dataset for sreg and ssreg
        self.data_sreg = self.create_dataset(src, pred, task="sreg")
        self.data_ssreg = self.create_dataset(src, pred, task="ssreg")

    def create_example(self, src, pred, task):
        examples = []
        if task == "ssreg":
            for i, (s, p) in enumerate(zip(src, pred)):
                examples.append(
                    InputExample(guid=i, text_a=s, text_b=p, label=None)
                )
        elif task == "sreg":
            for i, p in enumerate(pred):
                examples.append(
                    InputExample(guid=i, text_a=p, text_b=None, label=None)
                )
        return examples
    
    def convert_examples_to_features(
        self,
        examples,
        tokenizer,
        max_length=None,
        task=None,
        label_list=None,
        output_mode=None,
    ):
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample):
            if example.label is None:
                return None
            elif output_mode == "classification":
                return label_map[example.label]
            elif output_mode == "regression":
                return float(example.label)
            raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        return features

    def create_dataset(self, src, pred, task=None):
        # load examples and convert to features
        examples = self.create_example(src, pred, task=task)
        tokenizer = self.tokenizer
        features = self.convert_examples_to_features(
            examples,
            tokenizer,
            label_list=[None],
            max_length=512, # 各モデルで違うなら変える必要がある
            output_mode="regression",
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        return dataset

    def min_max_normalize(self, x, x_min=1, x_max=4):
        return (x - x_min) / (x_max - x_min)

    def score(self):
        # normalize
        g = self.predict(task="grammer")
        score_g = [self.min_max_normalize(x) for x in self.predict(task="grammer")]
        score_f = [self.min_max_normalize(x) for x in self.predict(task="fluency")]
        score_m = [self.min_max_normalize(x) for x in self.predict(task="meaning")]

        # calc gfm score
        final_score = []
        for g, f, m in zip(score_g, score_f, score_m):
            final_score.append(
                self.args.weight_g * g + self.args.weight_f * f + self.args.weight_m * m
            )

        return torch.tensor(final_score).reshape(-1, self.args.criterion_sample_size)

    def predict(self, task):
        # Setup CUDA, GPU & distributed training
        device = torch.device("cuda")

        if task == "grammer":
            model = self.model_g
            pred_dataset = self.data_sreg
        elif task == "fluency":
            model = self.model_f
            pred_dataset = self.data_sreg
        elif task == "meaning":
            model = self.model_m
            pred_dataset = self.data_ssreg

        model.to(device)

        pred_bach_size = self.args.bert_batch_size
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=pred_bach_size)

        preds = None

        for batch in pred_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1]}
                if self.args.model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = batch[2] if self.args.model_type in ['bert', 'xlnet'] else None  
                outputs = model(**inputs)
                logits = outputs[:2][0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        preds = np.squeeze(preds)
        return preds
