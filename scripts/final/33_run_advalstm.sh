#!/bin/bash
# official Adv-ALSTM (TF 1.15 conda env tf1), KDD17 hyper-parameters, Taiwan 50 walk-forward; CPU; saves models and re-scores the test set
# usage: 33_run_advalstm.sh <data_dir> <tag> <year>
cd /home/tobyleung/course/ds_project/external/Adv-ALSTM
PY=/home/tobyleung/miniconda3/envs/tf1/bin/python; D=$1; TAG=$2; yr=$3; R=/home/tobyleung/course/ds_project/results/final
export OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
export ADV_TRA=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['tra'])") ADV_VAL=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['val'])") ADV_TES=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['tes'])")
for M in LSTM ALSTM AdvALSTM; do mkdir -p saved_model/${TAG}_${yr}_${M}; done
$PY pred_lstm.py -p $D/$yr/ourpped/ -l 5 -u 4 -l2 0.001 -a 0 -f 1 -qs ./saved_model/${TAG}_${yr}_LSTM/model > $R/log_${TAG}_${yr}_LSTM.txt 2>&1
$PY pred_lstm.py -p $D/$yr/ourpped/ -l 5 -u 4 -l2 0.001 -a 0 -f 1 -o pred -rl 1 -q ./saved_model/${TAG}_${yr}_LSTM/model -qs ./saved_model/${TAG}_${yr}_LSTM/model > $R/log_${TAG}_${yr}_LSTM_test.txt 2>&1
$PY pred_lstm.py -p $D/$yr/ourpped/ -l 15 -u 16 -l2 0.001 -f 1 -qs ./saved_model/${TAG}_${yr}_ALSTM/model > $R/log_${TAG}_${yr}_ALSTM.txt 2>&1
$PY pred_lstm.py -p $D/$yr/ourpped/ -l 15 -u 16 -l2 0.001 -f 1 -o pred -rl 1 -q ./saved_model/${TAG}_${yr}_ALSTM/model -qs ./saved_model/${TAG}_${yr}_ALSTM/model > $R/log_${TAG}_${yr}_ALSTM_test.txt 2>&1
$PY pred_lstm.py -p $D/$yr/ourpped/ -l 15 -u 16 -l2 0.001 -v 1 -rl 1 -q ./saved_model/${TAG}_${yr}_ALSTM/model -la 0.05 -le 0.001 -f 1 -qs ./saved_model/${TAG}_${yr}_AdvALSTM/model > $R/log_${TAG}_${yr}_AdvALSTM.txt 2>&1
$PY pred_lstm.py -p $D/$yr/ourpped/ -l 15 -u 16 -l2 0.001 -v 1 -f 1 -o pred -rl 1 -q ./saved_model/${TAG}_${yr}_AdvALSTM/model -qs ./saved_model/${TAG}_${yr}_AdvALSTM/model > $R/log_${TAG}_${yr}_AdvALSTM_test.txt 2>&1
echo "$TAG $yr done"
