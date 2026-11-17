#!/bin/bash
# Official StockNet (Xu & Cohen, ACL 2018; external/stocknet-code, security-audited; Python-3 ports + MCC on test + per-run dirs, see 03 doc)
# on the official StockNet dataset (external/stocknet-dataset, 87 stocks with tweets), paper config.yml
# (alpha 0.5, 15 epochs, train 2014-01-01..2015-08-01, dev ..2015-10-01, test ..2016-01-01). The code has no seed: each run is a fresh draw.
# usage: 30_run_stocknet_official.sh <variant: hedge|tech|fund|discriminative> <run tag, e.g. r1> [device: cpu|0|1]
V=${1:-hedge}; RUN=${2:-r1}; DEV=${3:-cpu}
SRC=/home/tobyleung/course/ds_project/external/stocknet-code/src; R=/home/tobyleung/course/ds_project/results/final
W=/home/tobyleung/course/ds_project/external/stocknet-code/src_${V}_${RUN}; rm -rf $W; cp -r $SRC $W; cd $W   # private copy of src so config.yml edits don't race
if [ "$DEV" = "cpu" ]; then PY=/home/tobyleung/miniconda3/envs/tf1/bin/python; export CUDA_VISIBLE_DEVICES=""; else PY=/home/tobyleung/miniconda3/envs/tf1cuda/bin/python; export CUDA_VISIBLE_DEVICES=$DEV; fi
sed -i "s/^  variant_type: .*/  variant_type: '$V' # hedge, tech, fund, discriminative/" config.yml
export STOCKNET_RUN=_${V}_${RUN}; export STOCKNET_THREADS=${STOCKNET_THREADS:-12}
rm -rf ../checkpoints_${V}_${RUN} ../graphs_${V}_${RUN} ../log_${V}_${RUN}
$PY Main.py > $R/log_30_stocknet_official_${V}_${RUN}.txt 2>&1
echo "exit $? variant $V run $RUN device $DEV" >> $R/log_30_stocknet_official_${V}_${RUN}.txt
