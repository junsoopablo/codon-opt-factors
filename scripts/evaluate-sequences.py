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
from concurrent import futures
import inspect
import pandas as pd
import linearfold as lf
import linearpartition as lp
import RNA

class EvaluateSequences:

    def __init__(self, cds_file, utr_file, output_file):
        self.cds_file = cds_file
        self.utr_file = utr_file 
        self.output_file = output_file

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
                    'take_folding': 'folding' in inspect.getfullargspec(method).args
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

            for fold_type, folding in foldings.items():
                value = handler['handler'](name, seq, lengths, folding)
                yield (metric_name, fold_type, value)

    def fold(self, seq):
        foldings = {}

        fold, fe = lf.fold(seq)
        foldings['lf'] = {'structure': fold, 'free_energy': fe}

        foldings['lp'] = lp.partition(seq)

        fold, fe = RNA.fold(seq)
        foldings['v'] = {'structure': fold, 'free_energy': fe}

        return foldings

    def metric_free_energy(self, name, seq, lengths, folding):
        return folding['free_energy']


EvaluateSequences(snakemake.input.cds, snakemake.input.utr,
                  snakemake.output[0]).run(snakemake.threads)