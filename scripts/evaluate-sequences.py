#!/usr/bin/env python3
#
# Copyright (c) 2024 Seoul National University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# 

from Bio import SeqIO
from Bio.Data import CodonTable
from concurrent import futures
from vaxpress.data.codon_usage_data import codon_usage
from vaxpress.data.bicodon_usage_data import bicodon_usage
from itertools import product
from functools import partial
import numpy as np
import pandas as pd
import linearfold as lf
import linearpartition as lp
import RNA
import sys

sys.stdout, stdout_orig = open('/dev/null', 'w'), sys.stdout
import DegScore # suppress warning message during the import
sys.stdout = stdout_orig

def calc_single_codon_cai(codon_usage, codon_table):
    codon2aa = codon_table.forward_table.copy()
    for c in codon_table.stop_codons:
        codon2aa[c] = '*'

    tbl = pd.DataFrame({
        'freq': pd.Series(codon_usage),
        'aa': pd.Series(codon2aa)})

    log_weights = {}
    for aa, codons in tbl.groupby('aa'):
        log_weights.update(np.log2(codons['freq'] / codons['freq'].max()).to_dict())

    return log_weights

single_cai_weights = calc_single_codon_cai(codon_usage['Homo sapiens'],
                                           CodonTable.standard_rna_table)


def calc_bicodon_cai(codon_usage):
    pairs = [''.join(seq) for seq in product('ACGU', repeat=6)]
    log_weights = dict(zip(pairs, codon_usage))
    return log_weights

bicodon_cai_weights = calc_bicodon_cai(bicodon_usage['Homo sapiens'])


def calc_weighted_aup(__name, seq, lengths, folding, weights):
    seqbases = np.frombuffer(seq.encode(), dtype='i1')
    unpairprob_bypos = 1 - folding['bpp'].sum(axis=1).clip(0, 1)

    total = weightsum = 0.0
    for base, baseweight in weights.items():
        matches = (seqbases == ord(base))
        total += unpairprob_bypos[matches].sum() * baseweight
        weightsum += matches.sum() * baseweight

    return total / weightsum


class EvaluateSequences:

    def __init__(self, cds_file, utr_file, output_file):
        self.cds_file = cds_file
        self.utr_file = utr_file 
        self.output_file = output_file

        self.cache = {}
        self.metric_handlers = self.scan_metric_handlers()

        self.load_sequences(cds_file, utr_file)

    def load_sequences(self, cds_file, utr_file):
        utrseqs = SeqIO.to_dict(SeqIO.parse(utr_file, 'fasta'))

        self.fullseqs = {}
        self.cdsseqs = {}
        self.cdsinfo = {}
        self.seqnames = []

        for record in SeqIO.parse(cds_file, 'fasta'):
            utr5 = utr3 = None

            tags = record.description.split()[1:]
            for tag in tags:
                if tag.startswith('u5:'):
                    utr5 = utrseqs[tag].seq
                elif tag.startswith('u3:'):
                    utr3 = utrseqs[tag].seq
                else:
                    raise ValueError('Unknown tag:', tag)

            if utr5 is None or utr3 is None:
                raise ValueError('Missing UTR sequences:', record.description)

            name = record.name
            self.fullseqs[name] = str(utr5 + record.seq + utr3)
            self.cdsseqs[name] = str(record.seq)
            self.cdsinfo[name] = (len(utr5), len(record.seq), len(utr3))
            self.seqnames.append(name)

    def run(self, n_jobs):
        with open(self.output_file, 'w') as outf, \
                    futures.ProcessPoolExecutor(n_jobs) as executor:

            header_output = False
            for results in executor.map(self.calc_metrics, self.seqnames):
                results.to_csv(outf, sep='\t', index=False, header=not header_output)
                header_output = True

        print('Finished.')

    def scan_metric_handlers(self):
        handlers = {}
        for name in dir(self):
            if name.startswith('metric_') and callable(getattr(self, name)):
                method = getattr(self, name)
                handlers[name[len('metric_'):]] = {
                    'handler': method,
                    'take_folding': getattr(method, '__requires_folding__', False),
                    'blacklist': getattr(method, '__blacklist__', []),
                }
        return handlers

    def calc_metrics(self, name):
        cdsseq = self.cdsseqs[name]
        fullseq = self.fullseqs[name]
        utr5len, cdslen, utr3len = self.cdsinfo[name]

        seqvariants = [
            ['cdsonly', cdsseq, (0, cdslen, 0)],
            ['full', fullseq, (utr5len, cdslen, utr3len)],
        ]

        results = []
        for vname, seq, lengths in seqvariants:
            res = self.calc_metrics_for_seq(f'{name}:{vname}', seq, lengths)
            for metric_name, fold_type, value in res:
                results.append([name, vname, metric_name, fold_type, value])

        results = pd.DataFrame(results,
                               columns=['seqname', 'variant', 'metric', 'folder', 'value'])

        print('=> Finished:', name)

        return results

    def calc_metrics_for_seq(self, name, seq, lengths):
        print(f'-> Predicting secondary structures: {name}')

        foldings = self.fold(seq)

        for metric_name, handler in self.metric_handlers.items():
            if not handler['take_folding']:
                value = handler['handler'](name, seq, lengths)
                yield (metric_name, None, value)
                continue

            for fold_type, folding in foldings.items():
                if fold_type in handler['blacklist']:
                    continue

                value = handler['handler'](name, seq, lengths, folding)
                yield (metric_name, fold_type, value)

    def fold(self, seq):
        foldings = {}

        fold, fe = lf.fold(seq)
        foldings['LinearFold'] = {'structure': fold, 'free_energy': fe}

        foldings['LinearPartition'] = lp.partition(seq)
        # Convert list of non-zero probs to a matrix
        lpbpp = foldings['LinearPartition']['bpp']
        bpp = np.zeros((len(seq), len(seq)), dtype='f8')
        bpp[lpbpp['i'], lpbpp['j']] = lpbpp['prob']
        bpp += bpp.T
        foldings['LinearPartition']['bpp'] = bpp

        fold, fe = RNA.fold(seq)
        foldings['ViennaRNA:fold'] = {'structure': fold, 'free_energy': fe}

        fc = RNA.fold_compound(seq)
        fold, fe = fc.pf()
        bpp = np.array(fc.bpp())[1:, 1:]
        bpp += bpp.T
        foldings['ViennaRNA:partition'] = {
            'structure': fold.replace(',', '.').replace('{', '(').replace('}', ')').replace('|', '.'),
            'free_energy': fe,
            'bpp': bpp,
        }

        return foldings

    def metric_free_energy(self, name, seq, lengths, folding):
        return folding['free_energy']
    metric_free_energy.__requires_folding__ = True

    def metric_log2_single_cai(self, name, seq, lengths):
        cds = seq[lengths[0]:lengths[0]+lengths[1]]
        logwmean = np.mean([
            single_cai_weights[cds[i:i+3]] for i in range(0, len(cds), 3)])
        return logwmean

    def metric_log2_bicodon_cai(self, name, seq, lengths):
        cds = seq[lengths[0]:lengths[0]+lengths[1]]
        logwmean = np.mean([
            bicodon_cai_weights[cds[i:i+6]] for i in range(0, len(cds) - 3, 3)])
        return logwmean

    def metric_degscore(self, name, seq, lengths, folding):
        # This returns the original DegScore sum, not divided by sequence length
        return DegScore.DegScore(seq, structure=folding['structure']).degscore
    metric_degscore.__requires_folding__ = True
    metric_degscore.__blacklist__ = ['ViennaRNA:partition']

    #__metric_aup.__blacklist__ = ['LinearFold', 'ViennaRNA:fold']

    metric_aup_unweighted = partial(calc_weighted_aup, weights={
        'A': 1.0, 'C': 1.0, 'G': 1.0, 'U': 1.0})

    metric_aup_U3A2 = partial(calc_weighted_aup, weights={
        'A': 2.0, 'C': 1.0, 'G': 1.0, 'U': 3.0})

    metric_aup_U5A2 = partial(calc_weighted_aup, weights={
        'A': 2.0, 'C': 1.0, 'G': 1.0, 'U': 5.0})

    metric_aup_U3A1GC0 = partial(calc_weighted_aup, weights={
        'A': 1.0, 'C': 0.0, 'G': 0.0, 'U': 3.0})

    metric_aup_U1AGC0 = partial(calc_weighted_aup, weights={
        'A': 0.0, 'C': 0.0, 'G': 0.0, 'U': 1.0})

    for func in (metric_aup_unweighted, metric_aup_U3A2, metric_aup_U5A2,
                 metric_aup_U3A1GC0, metric_aup_U1AGC0):
        func.__requires_folding__ = True
        func.__blacklist__ = ['ViennaRNA:fold', 'LinearFold']

    def metric_unpaired_U(self, name, seq, lengths, folding):
        return sum((base == 'U' and pair == '.')
                   for base, pair in zip(seq, folding['structure']))
    metric_unpaired_U.__requires_folding__ = True

    def metric_unpaired_A(self, name, seq, lengths, folding):
        return sum((base == 'A' and pair == '.')
                   for base, pair in zip(seq, folding['structure']))
    metric_unpaired_A.__requires_folding__ = True

    def metric_unpaired_G(self, name, seq, lengths, folding):
        return sum((base == 'G' and pair == '.')
                   for base, pair in zip(seq, folding['structure']))
    metric_unpaired_G.__requires_folding__ = True

    def metric_unpaired_C(self, name, seq, lengths, folding):
        return sum((base == 'C' and pair == '.')
                   for base, pair in zip(seq, folding['structure']))
    metric_unpaired_C.__requires_folding__ = True

    def metric_total_loop_size(self, name, seq, lengths, folding):
        return sum(pair == '.' for pair in folding['structure'])
    metric_total_loop_size.__requires_folding__ = True


EvaluateSequences(snakemake.input.cds, snakemake.input.utr,
                  snakemake.output[0]).run(snakemake.threads)