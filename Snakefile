rule all:
    input:
        'tables/sequence-evaluations.tsv'

rule evaluate_sequences:
    input:
        cds='settings/CDS.fasta',
        utr='settings/UTR.fasta'
    output: 'tables/sequence-evaluations.tsv'
    threads: 32
    script: 'scripts/evaluate-sequences.py'
