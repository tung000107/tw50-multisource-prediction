#!/bin/bash
# Seed-variability check for the best published architecture on our data: official ALSTM (unchanged code and hyper-parameters) on the
# nine-source inputs, re-run with a free random initialisation (-f 0; the first run used -f 1 = fixed TF seed 123456).
# usage: 38_run_alstm_reruns.sh <rerun_id> <year>      -> saved_model/33_adv74r<id>_<year>_ALSTM/, results/final/log_33_adv74r<id>_<year>_ALSTM.txt
cd /home/tobyleung/course/ds_project/external/Adv-ALSTM
PY=/home/tobyleung/miniconda3/envs/tf1/bin/python; D=/home/tobyleung/course/ds_project/data/raw/tw50_advalstm74; TAG=33_adv74r$1; yr=$2; R=/home/tobyleung/course/ds_project/results/final
export OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
export ADV_TRA=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['tra'])") ADV_VAL=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['val'])") ADV_TES=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['tes'])")
mkdir -p saved_model/${TAG}_${yr}_ALSTM
$PY pred_lstm.py -p $D/$yr/ourpped/ -l 15 -u 16 -l2 0.001 -f 0 -qs ./saved_model/${TAG}_${yr}_ALSTM/model > $R/log_${TAG}_${yr}_ALSTM.txt 2>&1
$PY pred_lstm.py -p $D/$yr/ourpped/ -l 15 -u 16 -l2 0.001 -f 0 -o pred -rl 1 -q ./saved_model/${TAG}_${yr}_ALSTM/model -qs ./saved_model/${TAG}_${yr}_ALSTM/model > $R/log_${TAG}_${yr}_ALSTM_test.txt 2>&1
echo "$TAG $yr done"
