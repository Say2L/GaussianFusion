TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=gaussianfusion_agent \
experiment_name=training_gaussianfusion_agent \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
force_cache_computation=False