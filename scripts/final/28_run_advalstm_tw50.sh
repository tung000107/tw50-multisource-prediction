#!/bin/bash
# official Adv-ALSTM (TF 1.15 conda env tf1), KDD17 hyper-parameters, Taiwan 50 walk-forward; CPU
cd /home/tobyleung/course/ds_project/external/Adv-ALSTM
PY=/home/tobyleung/miniconda3/envs/tf1/bin/python; D=/home/tobyleung/course/ds_project/data/raw/tw50_advalstm; R=/home/tobyleung/course/ds_project/results/final
for yr in 2023 2024 2025 2026; do
  export ADV_TRA=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['tra'])") ADV_VAL=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['val'])") ADV_TES=$(python3 -c "import json;print(json.load(open('$D/meta.json'))['$yr']['tes'])")
  $PY pred_lstm.py -p $D/$yr/ourpped/ -l 5 -u 4 -l2 0.001 -a 0 -f 1 > $R/log_28_adv_${yr}_LSTM.txt 2>&1
  mkdir -p saved_model/tw50_${yr}_alstm; $PY pred_lstm.py -p $D/$yr/ourpped/ -l 15 -u 16 -l2 0.001 -f 1 -qs ./saved_model/tw50_${yr}_alstm/model > $R/log_28_adv_${yr}_ALSTM.txt 2>&1
  $PY pred_lstm.py -p $D/$yr/ourpped/ -l 15 -u 16 -l2 0.001 -v 1 -rl 1 -q ./saved_model/tw50_${yr}_alstm/model -la 0.05 -le 0.001 -f 1 > $R/log_28_adv_${yr}_AdvALSTM.txt 2>&1
  echo "year $yr done"
done
echo ALLDONE
