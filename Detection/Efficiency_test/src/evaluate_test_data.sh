python evaluate_test_data.py \
--trigger-threshold 31 \
--injection-file /workspace/ligo_data/ml-training-strategies/TestData/injections.hdf \
--data-dir /workspace/ligo_data/ml-training-strategies/TestData/output \
--start-time-offset 0.7 \
--test-data-activation linear \
--ranking-statistic linear \
--delta-t 0.099609375 \
--trigger-file-name triggers_linear_3.hdf \
--event-file-name events_linear_3.hdf \
--stats-file-name stats_linear_3.hdf \
--verbose