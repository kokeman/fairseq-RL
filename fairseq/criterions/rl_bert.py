# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from fairseq import metrics, utils, search
from fairseq.criterions import FairseqCriterion, register_criterion

from fairseq.sequence_generator import SequenceGenerator
from fairseq import bleu, bert

from mosestokenizer import MosesDetokenizer

@register_criterion('rl_bert')
class RLCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.n_sample = args.criterion_sample_size
        self.ce_weight = args.ce_weight
        self.pad = task.tgt_dict.pad()
        search_strategy = search.Sampling(task.tgt_dict) if args.search_strategy == "sampling" else None
        self.sample_gen = SequenceGenerator(task.tgt_dict, beam_size=self.n_sample, retain_dropout=False, search_strategy=search_strategy)
        self.greedy_gen = SequenceGenerator(task.tgt_dict, beam_size=1, retain_dropout=False)
        # self.scorer = bleu.Scorer(task.tgt_dict.pad(), task.tgt_dict.eos(), task.tgt_dict.unk(), args.sent_bleu)
        self.scorer = bert.Scorer(args)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, reward = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'reward': reward[0],
            'baseline_reward': reward[1],
            'loss': loss[0].data,
            'ce_loss': loss[1].data,
            'rl_loss': loss[2].data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss[0], sample_size, logging_output

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--criterion-sample-size', type=int, default=5, help='Number of sample size (default: 5)')
        parser.add_argument('--ce-weight', type=float, default=0.1, help='Weight of cross entropy loss (default: 0.1)')
        parser.add_argument('--baseline-reward', default="average", help='The method of baseline reward (average or self-critic)')
        parser.add_argument('--reward-clipping', action='store_true', default=False, help='if rewad is negative, reward is set to 0')
        parser.add_argument('--sent-bleu', action='store_true', default=False, help='if true, the reward bleu is normalized 0 ~ 1')
        parser.add_argument('--search-strategy', default="beam", help='sampling strategy (sampling or beam)')
        parser.add_argument('--mixier', action='store_true', default=False, 
                                help='Linearly reduce the ce-weight from --ce-weith to --min-ce-weight')
        parser.add_argument('--min-ce-weight', type=float, default=0.03)
        parser.add_argument('--rl-epoch-num', type=int, default=10, help="epoch num on rl")
        parser.add_argument('--bert-dir', help="fine-tuned bert model dir")
        parser.add_argument('--g-dir', help="fine-tuned grammer model dir")
        parser.add_argument('--f-dir', help="fine-tuned fluency model dir")
        parser.add_argument('--m-dir', help="fine-tuned meaning model dir")
        parser.add_argument('--weight-g', type=float, default=0.2, help="weight of grammer score")
        parser.add_argument('--weight-f', type=float, default=0.5, help="weight of fluency score")
        parser.add_argument('--weight-m', type=float, default=0.3, help="weight of meaning score")
        parser.add_argument('--bert-batch-size', type=int, default="16")
        parser.add_argument('--model-type', default="bert", help="model type")

    def reword(self, ref, pred):
        self.scorer.reset(one_init=True)
        self.scorer.add(ref.type(torch.IntTensor), pred.type(torch.IntTensor))
        return self.scorer.score()
    
    def bert_reward(self, src, pred):
        self.scorer.add(src, pred)
        return self.scorer.score()
        
    def remove_bpe(self, line, bpe_symbol):
        line = line.replace("\n", '')
        line = (line + ' ').replace(bpe_symbol, '').rstrip()
        return line+("\n")

    def compute_loss(self, model, net_output, sample, reduce=True):
        ce_loss, _ = self.compute_ce_loss(model, net_output, sample, reduce=reduce)
        rl_loss, reward = self.compute_rl_loss(model, net_output, sample, reduce=reduce)

        if self.args.mixier:
            epoch = metrics.get_meter("global", "epoch").val
            alpha = epoch * (self.args.min_ce_weight - self.args.ce_weight) / (self.args.rl_epoch_num - 1) + self.args.ce_weight
        else:
            alpha = self.ce_weight
        metrics.log_scalar('alpha', alpha)

        loss = alpha * ce_loss + (1.0 - alpha) * rl_loss

        return [loss, ce_loss, rl_loss], reward

    def compute_rl_loss(self, model, net_output, sample, reduce=True):
        # Generate baseline/samples
        y_hat = self.sample_gen.generate([model], sample)
        ref = sample['target']

        ### rewords ###
        y_hat_ = [y_hat_i_n['tokens'] for y_hat_i in y_hat for y_hat_i_n in y_hat_i]
        y_hat_ = [" ".join([self.task.tgt_dict[t_i] for t_i in t]) for t in y_hat_]
        src = [" ".join([self.task.src_dict[t_i] for t_i in t]) for t in sample['net_input']['src_tokens']]
        # remove bpe
        y_hat_ = [self.remove_bpe(s, '@@ ') for s in y_hat_]
        src = [self.remove_bpe(s, '@@ ') for s in src]
        # detokenize
        with MosesDetokenizer('en') as detokenize:
            y_hat_ = [detokenize(s.split()) for s in y_hat_]
            src = [detokenize(s.split()) for s in src]
        # Aligh src and pred
        src = [s for s in src for _ in range(self.n_sample)]
        # r_hat = torch.tensor([[self.reword(ref_i, y_hat_i_n['tokens']) for y_hat_i_n in y_hat_i] for ref_i, y_hat_i in zip(ref, y_hat)])
        r_hat = torch.tensor(self.bert_reward(y_hat_, src))

        if self.args.baseline_reward == "average":
            r_b = r_hat.mean(1, True)
        elif self.args.baseline_reward == "self-critic":
            y_g = self.greedy_gen.generate([model], sample)
            r_b = torch.tensor([self.reword(ref_i, y_g_i[0]['tokens']) for ref_i, y_g_i in zip(ref, y_g)]).unsqueeze(-1)

        r_d = r_hat - r_b

        if self.args.reward_clipping:
            r_d[r_d < 0] = 0

        # scores
        net_input = {
            'src_tokens': sample['net_input']['src_tokens'],
            'src_lengths': sample['net_input']['src_lengths'],
        }
        encoder_out = model.encoder(**net_input)
        bos = sample['net_input']['prev_output_tokens'][:,:1]

        scores = []
        for n in range(self.n_sample):
            output_tokens = [y_hat_i[n]['tokens'] for y_hat_i in y_hat]
            output_tokens = rnn_utils.pad_sequence(output_tokens, batch_first=True, padding_value=self.pad)

            prev_output_tokens = torch.cat([bos, output_tokens], dim=-1)
            net_output = model.decoder(prev_output_tokens, encoder_out=encoder_out)

            lprobs = model.get_normalized_probs(net_output, log_probs=True)[:, :-1, :]  # (batch_size, length, vocab_size)
            lprobs = lprobs.reshape(-1, lprobs.size(-1))  # (batch_size * length, vocab_size)
            lprobs = lprobs[range(lprobs.size(0)), output_tokens.reshape(-1)]  # (batch_size * length)
            lprobs = lprobs.reshape(output_tokens.size())  # (batch_size, length)
            lprobs = lprobs.sum(dim=-1, keepdim=True)  # (batch_size, 1)

            scores.append(lprobs)

        scores = torch.cat(scores, dim=-1)  # (batch_size, sample_size)
        r_d = r_d.to(scores.device)  # (batch_size, sample_size)

        loss = ((-scores * r_d) / self.n_sample).sum()

        return loss, [r_hat.mean(), r_b.mean()]

    def compute_ce_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ce_loss_sum = utils.item(sum(log.get('ce_loss', 0) for log in logging_outputs))
        rl_loss_sum = utils.item(sum(log.get('rl_loss', 0) for log in logging_outputs))
        reward_sum = utils.item(sum(log.get('reward', 0) for log in logging_outputs))
        baseline_reward_sum = utils.item(sum(log.get('baseline_reward', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ce_loss', ce_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('rl_loss', rl_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('reward', reward_sum, round=3)
        metrics.log_scalar('baseline_reward', baseline_reward_sum, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
