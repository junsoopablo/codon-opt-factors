DEGSCORE_RAW_PREFIX = 'https://raw.githubusercontent.com/eternagame/DegScore/master/'
DEGSCORE_PY_URL = f'{DEGSCORE_RAW_PREFIX}DegScore.py'
ASSIGN_LOOP_TYPE_PY_URL = f'{DEGSCORE_RAW_PREFIX}assign_loop_type.py'

rule all:
    input:
        'tables/sequence-evaluations.tsv'

rule evaluate_sequences:
    input:
        cds='settings/CDS.fasta',
        utr='settings/UTR.fasta',
        degscore_py='scripts/DegScore.py',
        assign_loop_type_py='scripts/assign_loop_type.py'
    output: 'tables/sequence-evaluations.tsv'
    threads: 32
    script: 'scripts/evaluate-sequences.py'

rule download_degscore:
    output:
        degscore_py='scripts/DegScore.py',
        assign_loop_type_py='scripts/assign_loop_type.py'
    shell:
        'wget -O {output.degscore_py} {DEGSCORE_PY_URL} && '
        'wget -O {output.assign_loop_type_py} {ASSIGN_LOOP_TYPE_PY_URL}'
