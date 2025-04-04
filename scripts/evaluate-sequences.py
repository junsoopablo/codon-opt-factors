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
from codon_usage_data import codon_usage
from bicodon_usage_data import bicodon_usage
from itertools import product
from functools import partial
import numpy as np
import pandas as pd
import linearfold as lf
import linearpartition as lp
import warnings
import re
import os
import RNA
import sys

sys.stdout, stdout_orig = open('/dev/null', 'w'), sys.stdout
import DegScore # suppress warning message during the import
sys.stdout = stdout_orig

try:
    import rpy2.robjects.packages as rpackages
    rpackages.importr('iCodon')
    rpackages.importr('stringr')
except (ModuleNotFoundError, rpackages.PackageNotInstalledError):
    iCodon_installed = False
    warnings.warn('iCodon R package not found. '
                  'Try: mamba install -c ChangLabSNU -c bioconda r-icodon-light rpy2')
else:
    iCodon_installed = True

    import rpy2.robjects as ro
    os.environ['TZ'] = 'UTC' # dplyr requires this to run in singularity
    ro.r['options'](warn=-1)
    iCodon_predict_stability = ro.r['predict_stability']('human')


FOLDER_VIENNARNA_FOLD = 'ViennaRNA:fold'
FOLDER_VIENNARNA_PARTITION = 'ViennaRNA:partition'
FOLDER_LINEARFOLD = 'LinearFold'
FOLDER_LINEARPARTITION = 'LinearPartition'

CAPABILITY_FOLDING = 'folding'
CAPABILITY_FOLDING_STRICT = 'folding_strict' # produces perfectly paired structure
CAPABILITY_BPP_MATRIX = 'bpp_matrix'

folder_capabilities = {
    FOLDER_VIENNARNA_FOLD: {
        CAPABILITY_FOLDING,
        CAPABILITY_FOLDING_STRICT,
    },
    FOLDER_VIENNARNA_PARTITION: {
        CAPABILITY_FOLDING,
        CAPABILITY_BPP_MATRIX,
    },
    FOLDER_LINEARFOLD: {
        CAPABILITY_FOLDING,
        CAPABILITY_FOLDING_STRICT,
    },
    FOLDER_LINEARPARTITION: {
        CAPABILITY_FOLDING,
        CAPABILITY_FOLDING_STRICT,
        CAPABILITY_BPP_MATRIX,
    },
}


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

def calc_pairing(structure):
    stack = []
    basepairs = []
    heightmap = []

    for i, stype in enumerate(structure):
        heightmap.append(len(stack))

        if stype == '.':
            continue
        elif stype == '(':
            stack.append(i)
            heightmap[-1] += 1
        elif stype == ')':
            if stack:
                j = stack.pop()
                basepairs.append((j, i))
            else:
                raise ValueError(f'Unpaired base pair closure at position {i+1}')
        else:
            raise ValueError('Unknown structure annotation: ' + stype)

    if stack:
        raise ValueError(f'Unpaired base pair closure at position {stack[0]+1}')

    # Build a pair position map
    pairmap = [None] * len(structure)
    for i, j in basepairs:
        pairmap[i] = j
        pairmap[j] = i

    return {
        'base_pairs': sorted(basepairs),
        'pair_map': pairmap,
        'height_map': heightmap,
    }

def calc_max_stem_length(structure, pairmap, minimum_length=3):
    for cand_match in re.finditer(r'\({' + str(minimum_length) + ',}', structure):
        cand_start, cand_end = cand_match.span()
        cand_end -= 1 # make it inclusive

        pair_start = pairmap[cand_start]
        pair_end = pairmap[cand_end]

        stem_start = pair_end
        maxlen = 1
        for i in range(pair_end + 1, pair_start + 1):
            if structure[i] in '.(':
                stem_start = None
                continue

            if stem_start is None:
                stem_start = i
            else:
                length = i - stem_start + 1
                if length > maxlen:
                    maxlen = length

        return maxlen


def requires(*capabilities):
    def decorator(func):
        func.__requires__ = set(capabilities)
        return func
    return decorator


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
                    'requires': getattr(method, '__requires__', set()),
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

        return results

    def calc_metrics_for_seq(self, name, seq, lengths):
        print(f'-> Predicting secondary structures: {name}')
        foldings = self.fold(seq)

        print(f'-> Calculating metrics: {name}')
        for metric_name, handler in self.metric_handlers.items():
            requirements = handler['requires']

            if CAPABILITY_FOLDING not in requirements:
                value = handler['handler'](name, seq, lengths)
                yield (metric_name, None, value)
                continue

            for fold_type, folding in foldings.items():
                capabilities = folder_capabilities[fold_type]
                if len(requirements & capabilities) != len(requirements):
                    continue

                value = handler['handler'](name, seq, lengths, folding)
                yield (metric_name, fold_type, value)

    def fold(self, seq):
        foldings = {}

        fold, fe = lf.fold(seq)
        foldings['LinearFold'] = {'structure': fold, 'free_energy': fe}
        foldings['LinearFold'].update(calc_pairing(fold))

        foldings['LinearPartition'] = lp.partition(seq)
        # Convert list of non-zero probs to a matrix
        lpbpp = foldings['LinearPartition']['bpp']
        bpp = np.zeros((len(seq), len(seq)), dtype='f8')
        bpp[lpbpp['i'], lpbpp['j']] = lpbpp['prob']
        bpp += bpp.T
        foldings['LinearPartition']['bpp'] = bpp
        foldings['LinearPartition'].update(
            calc_pairing(foldings['LinearPartition']['structure']))

        fold, fe = RNA.fold(seq)
        foldings['ViennaRNA:fold'] = {'structure': fold, 'free_energy': fe}
        foldings['ViennaRNA:fold'].update(calc_pairing(fold))

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


    ## Simple secondary structure metrics

    @requires(CAPABILITY_FOLDING)
    def metric_free_energy(self, name, seq, lengths, folding):
        return folding['free_energy']


    ## Codon usage metrics

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


    ## DegScore metrics

    @requires(CAPABILITY_FOLDING, CAPABILITY_FOLDING_STRICT)
    def metric_degscore(self, name, seq, lengths, folding):
        # This returns the original DegScore sum, not divided by sequence length
        return DegScore.DegScore(seq, structure=folding['structure']).degscore


    ## Average Unpaired Probability (AUP) metrics

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
        func.__requires__ = {CAPABILITY_FOLDING, CAPABILITY_BPP_MATRIX}


    # Unpaired base type metrics

    @requires(CAPABILITY_FOLDING)
    def metric_unpaired_U(self, name, seq, lengths, folding):
        return sum((base == 'U' and pair == '.')
                   for base, pair in zip(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_unpaired_A(self, name, seq, lengths, folding):
        return sum((base == 'A' and pair == '.')
                   for base, pair in zip(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_unpaired_G(self, name, seq, lengths, folding):
        return sum((base == 'G' and pair == '.')
                   for base, pair in zip(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_unpaired_C(self, name, seq, lengths, folding):
        return sum((base == 'C' and pair == '.')
                   for base, pair in zip(seq, folding['structure']))


    ## Secondary structure categorization metrics

    @requires(CAPABILITY_FOLDING)
    def metric_total_loop_length(self, name, seq, lengths, folding):
        return sum(pair == '.' for pair in folding['structure'])

    @requires(CAPABILITY_FOLDING, CAPABILITY_FOLDING_STRICT)
    def metric_max_stem_length(self, name, seq, lengths, folding):
        return calc_max_stem_length(folding['structure'], folding['pair_map'])

    @requires(CAPABILITY_FOLDING)
    def metric_total_hairpin_loop_length(self, name, seq, lengths, folding):
        return sum(map(len, re.findall(r'\((\.+)\)', folding['structure'])))

    @requires(CAPABILITY_FOLDING)
    def metric_total_hairpin_loop_count(self, name, seq, lengths, folding):
        return len(re.findall(r'\(\.+\)', folding['structure']))

    @staticmethod
    def find_hairpin_loop_sequences(seq, structure):
        for matches in re.finditer(r'\((\.+)\)', structure):
            yield seq[slice(*matches.span(1))]

    @requires(CAPABILITY_FOLDING)
    def metric_total_hairpin_loop_U(self, name, seq, lengths, folding):
        return sum(seq.count('U') for seq
                   in self.find_hairpin_loop_sequences(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_total_hairpin_loop_A(self, name, seq, lengths, folding):
        return sum(seq.count('A') for seq
                   in self.find_hairpin_loop_sequences(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_total_hairpin_loop_G(self, name, seq, lengths, folding):
        return sum(seq.count('G') for seq
                   in self.find_hairpin_loop_sequences(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_total_hairpin_loop_C(self, name, seq, lengths, folding):
        return sum(seq.count('C') for seq
                   in self.find_hairpin_loop_sequences(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_total_bulge_length(self, name, seq, lengths, folding):
        return sum(
            len(lmatch) + len(rmatch) for lmatch, rmatch
            in re.findall(r'\((\.+)\(|\)(\.+)\)', folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_total_bulge_count(self, name, seq, lengths, folding):
        return len(re.findall(r'\((\.+)\(|\)(\.+)\)', folding['structure']))

    @staticmethod
    def find_bulge_sequences(seq, structure):
        for matches in re.finditer(r'\((\.+)\(|\)(\.+)\)', structure):
            span = matches.span(1)
            if span[0] < 0:
                span = matches.span(2)
            yield seq[slice(*span)]

    @requires(CAPABILITY_FOLDING)
    def metric_total_bulge_U(self, name, seq, lengths, folding):
        return sum(seq.count('U') for seq
                   in self.find_bulge_sequences(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_total_bulge_A(self, name, seq, lengths, folding):
        return sum(seq.count('A') for seq
                   in self.find_bulge_sequences(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_total_bulge_G(self, name, seq, lengths, folding):
        return sum(seq.count('G') for seq
                   in self.find_bulge_sequences(seq, folding['structure']))

    @requires(CAPABILITY_FOLDING)
    def metric_total_bulge_C(self, name, seq, lengths, folding):
        return sum(seq.count('C') for seq
                   in self.find_bulge_sequences(seq, folding['structure']))


    ## Base count metrics

    def metric_total_basecount_U(self, name, seq, lengths):
        return sum(base == 'U' for base in seq)

    def metric_total_basecount_A(self, name, seq, lengths):
        return sum(base == 'A' for base in seq)

    def metric_total_basecount_G(self, name, seq, lengths):
        return sum(base == 'G' for base in seq)

    def metric_total_basecount_C(self, name, seq, lengths):
        return sum(base == 'C' for base in seq)

    if iCodon_installed:
        def metric_icodon_stability(self, name, seq, lengths):
            cds = seq[lengths[0]:lengths[0]+lengths[1]]
            return float(iCodon_predict_stability([cds])[0])


EvaluateSequences(snakemake.input.cds, snakemake.input.utr,
                  snakemake.output[0]).run(snakemake.threads)
