#!/bin/bash
# Full rerun after the 2026-11-17 feature fixes (U.S. holiday join in build_dataset.py, TX night-return contract roll in 18, news text-score purge in 15).
# Order follows the data dependencies; every step logs "name exit code" to results/final/log_rerun_1117.txt and the chain stops on the first failure.
cd /home/tobyleung/course/ds_project; PY=.venv/bin/python; L=results/final/log_rerun_1117.txt; F=scripts/final; export CUDA_VISIBLE_DEVICES=0
step() { local n="$1"; shift; local t0=$(date +%s); "$@" > results/final/log_rerun_1117_${n}.txt 2>&1; local rc=$?; echo "$(date +%H:%M) $n exit $rc ($(( $(date +%s) - t0 ))s)" >> $L; [ $rc -eq 0 ] || { echo "CHAIN STOPPED at $n" >> $L; exit 1; }; }
echo "$(date +%H:%M) START" > $L
step build_dataset $PY scripts/build_dataset.py
step 01_prepare $PY $F/01_prepare.py
step 18_extra $PY $F/18_extra_features.py
step 15_news $PY $F/15_news_features.py
step 02_tabular $PY $F/02_tabular.py
step 03_sequence $PY $F/03_sequence.py
step 06_labels $PY $F/06_labels.py
step 04_stats $PY $F/04_stats.py
step 05_audits $PY $F/05_audits.py
step 09_explain $PY $F/09_explain.py
step 10_backtest $PY $F/10_backtest.py
for s in 16_news_experiment 19_source_study 29_review_fixes 22_graph_features 23_gap_fills 24_more_fills 25_news_analysis 26_supply_chain 27_llm_news_lookahead; do step $s $PY $F/$s.py; done
for s in lgb seq causal causal_tuned adv_gen; do step 33_$s $PY $F/33_sota_headtohead.py $s; done
step 33_adv_gen_all9 $PY $F/33_sota_headtohead.py adv_gen all9
# official ALSTM / Adv-ALSTM (TF1, CPU): 45- and 74-feature inputs, 4 test years, 4 jobs at a time; then the three free-init ALSTM reruns on 74 features
alstm() { local t0=$(date +%s); for yr in 2023 2024 2025 2026; do bash $F/33_run_advalstm.sh $PWD/data/raw/tw50_advalstm45 33_adv45 $yr & done; wait
          for yr in 2023 2024 2025 2026; do bash $F/33_run_advalstm.sh $PWD/data/raw/tw50_advalstm74 33_adv74 $yr & done; wait
          for r in 1 2 3; do for yr in 2023 2024 2025 2026; do bash $F/38_run_alstm_reruns.sh $r $yr & done; wait; done; }
step alstm_official alstm
step 33_adv_collect $PY $F/33_sota_headtohead.py adv_collect
step 33_assemble $PY $F/33_sota_headtohead.py assemble
step 34_final_checks $PY $F/34_final_model_checks.py
for s in build extra score; do step 35_$s $PY $F/35_fresh_holdout.py $s; done
for s in 36_alignment_audit 37_night_session_analysis; do step $s $PY $F/$s.py; done
for s in main more more2 bt alstm; do step 38_$s $PY $F/38_robustness.py $s; done
for s in 39_four_level_deep 39c_direction_size_models 41_eda_and_errors 42_external_eda 43_example_ranking 44_insight_checks 11_figures 34b_final_model_figure 39b_four_level_figure 40_multiseed_figures 25b_news_figure 31_diagram_ipo 13_full_report; do step $s $PY $F/$s.py; done
step deck bash -c "cd final_present/deck && bun build.js"
echo "$(date +%H:%M) CHAIN_DONE" >> $L
