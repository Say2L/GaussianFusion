TRAIN_TEST_SPLIT=navtest
CHECKPOINT=''

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=gaussianfusion_agent \
worker=single_machine_thread_pool \
experiment_name=gaussianfusion_agent_eval