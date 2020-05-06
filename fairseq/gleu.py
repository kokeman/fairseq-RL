from math import exp
from operator import mul
from collections import defaultdict
from functools import reduce

import math
from collections import Counter

class SentenceGleu():
    """
    Smoothed sentence-level GLEU as proposed by 
    Napoles, Sakaguchi, Post, and Tetreault (2015). 

    This is modified version by Yoshimura
    This resulting score is the same as https://github.com/cnap/gec-ranking/
    """

    def __init__(self, args):
        self.n = 4
        self.args = args

    def add(self, source_tokens, hypothesis_tokens, reference_tokens):
        self._source_length = len(source_tokens)
        self._hypothesis_length = len(hypothesis_tokens)
        self._reference_length = len(reference_tokens)
        self._source_ngrams = self._get_ngrams(source_tokens, self.n)
        self._hypothesis_ngrams = self._get_ngrams(hypothesis_tokens, self.n)
        self._reference_ngrams = self._get_ngrams(reference_tokens, self.n)

    def _get_ngrams(self, tokens, max_n):
        """
        Extracts all n-grams of order 1 up to (and including) @param max_n from
        a list of @param tokens.
        """
        n_grams = []
        for n in range(1, max_n + 1):
            n_grams.append(defaultdict(int))
            for n_gram in zip(*[tokens[i:] for i in range(n)]):
                n_grams[n - 1][n_gram] += 1
        return n_grams

    def score(self):
        """
        Scores @param hypothesis against this reference.
        @return the smoothed sentence-level BLEU score: 1.0 is best, 0.0 worst.
        """
        def product(iterable):
            return reduce(mul, iterable, 1)

        def ngram_precisions(src_ngrams, ref_ngrams, hyp_ngrams):
            precisions = []
            # for n in [4]:
            for n in range(1, self.n + 1):
                overlap = 0
                for ref_ngram, ref_ngram_count in ref_ngrams[n - 1].items():
                    if ref_ngram in hyp_ngrams[n - 1]:
                        overlap += min(ref_ngram_count, hyp_ngrams[n - 1][ref_ngram])

                # add penalty between source-only n-grams and hypothesis n-grams
                # # 1. get src-ref diff ngrams
                # for src_ngram, src_ngram_count in src_ngrams[n - 1].items():
                #     if src_ngram in ref_ngrams[n - 1]:
                #         src_ngram_count = max(0, src_ngram_count - ref_ngrams[n - 1][src_ngram])
                # # 2. compute penalty ngrams
                # penalty = 0
                # for src_ngram, src_ngram_count in src_ngrams[n - 1].items():
                #     if src_ngram in hyp_ngrams[n - 1]:
                #         penalty += min(src_ngram_count, hyp_ngrams[n - 1][src_ngram])

                # modified (penalty is the overlap n-gram count 
                #           not included in the reference but included in the source)
                penalty = 0
                for hyp_ngram, hyp_ngram_count in hyp_ngrams[n - 1].items():
                    if hyp_ngram in src_ngrams[n - 1] and hyp_ngram not in ref_ngrams[n - 1]:
                        penalty += min(hyp_ngram_count, src_ngrams[n - 1][hyp_ngram])

                hyp_length = max(0, self._hypothesis_length - n + 1)
                # if n >= 2:
                #     # smoothing as proposed by Lin and Och (2004),
                #     # implemented as described in (Chen and Cherry, 2014)
                #     overlap += 1
                #     hyp_length += 1

                # sentence level smooth
                numerator = max(overlap - penalty, 0)
                numerator = numerator if numerator != 0 else 1
                hyp_length = hyp_length if hyp_length != 0 else 1

                # precisions.append(max(0, (overlap - penalty) / hyp_length) if hyp_length > 0 else 0.0)
                precisions.append(max(0, numerator / hyp_length) if hyp_length > 0 else 0.0)

            return precisions

        def brevity_penalty(ref_length, hyp_length):
            return min(1.0, exp(1 - (ref_length / hyp_length if hyp_length > 0 else 0.0)))

        # calculate n-gram precision for all orders
        np = ngram_precisions(self._source_ngrams, self._reference_ngrams, self._hypothesis_ngrams)
        # calculate brevity penalty
        bp = brevity_penalty(self._reference_length, self._hypothesis_length)
        # compose final GLEU score
        return product(np)**(1 / self.n) * bp


def main():
    # src = [l.rstrip().split() for l in open("/clwork/yoshimura/data/corpus/conll/conll2014.src")]
    # hyp = [l.rstrip().split() for l in open("/clwork/yoshimura/evaluation/data/original/official_submissions/INPUT")]
    # ref = [l.rstrip().split() for l in open("/clwork/yoshimura/GEC/grammaticality-metrics/references/BN1")]

    # scorer = SentenceGleu()
    # result = []
    # for s, h, r in zip(src, hyp, ref):
    #     scorer.add(s, h, r)
    #     result.append(scorer.score())
    # with open("INPUT_mygleu", "w") as f:
    #     for r in result:
    #         f.write(f"{r:.6f}\n")
    #         # f.write(f"{r}\n")

    scorer = SentenceGleu()
    src = "Fish firming uses the lots of special products such as fish meal ."
    hyp = "Fish contains a lot of special products such as fish meals ."
    ref = "Fish firming uses a lot of special products such as fish meal ."
    scorer.add(src.split(), hyp.split(), ref.split())
    print(scorer.score())

if __name__ == "__main__":
    main()
