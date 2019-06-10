

PYTHONPATH=src python src/scripts/fever_gplsi/pipeline.py \
    --in_file $1 \
    --out_file /tmp/ir.$(basename $1)