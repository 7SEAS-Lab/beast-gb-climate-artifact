#!/usr/bin/env bash
export NNODES=${1:-4}
export BEAST_DL_DSYNC_INIT=1
export BEAST_USE_TE_DPA=1

# UP 1
# sbatch -A your_account -p booster --nodes $NNODES --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=up-scaling-$NNODES experiments/run_gb_script.sh systems/jupiter_container.sh src/main.py  --skip_validation --config_file configs/hero_config_jupiter.yaml mesh_dims=1,-1,4,2 data_train_subset=50 training_n_epochs=2 training_log_interval=5

# UP 16
# sbatch -A your_account -p booster --nodes $NNODES --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=up-scaling-$NNODES experiments/run_gb_script.sh systems/jupiter_container.sh src/main.py  --skip_validation --config_file configs/hero_config_jupiter.yaml mesh_dims=16,-1,4,2 data_train_subset=800 training_n_epochs=2 training_log_interval=5

# DP 2
# sbatch -A your_account -p booster --nodes $NNODES --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=dp-scaling-$NNODES experiments/run_gb_script.sh systems/jupiter_container.sh src/main.py  --skip_validation --config_file configs/hero_config_jupiter.yaml mesh_dims=-1,2,4,2 data_train_subset=$(( 50 * $NNODES / 4 )) training_n_epochs=2 training_log_interval=5

# DP 16
# sbatch -A your_account -p booster --nodes $NNODES --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=dp-scaling-$NNODES experiments/run_gb_script.sh systems/jupiter_container.sh src/main.py  --skip_validation --config_file configs/hero_config_jupiter.yaml mesh_dims=-1,16,4,2 data_train_subset=$(( 25 * $NNODES / 16 )) training_n_epochs=2 training_log_interval=5

# Predictive DP 4
# sbatch -A your_account -p booster --nodes $NNODES --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=pred-dp-scaling-$NNODES experiments/run_gb_script.sh systems/jupiter_container.sh src/main.py  --skip_validation --config_file configs/predictive_config_jupiter.yaml mesh_dims=4,-1,4,2 data_train_subset=200 training_n_epochs=2 training_log_interval=5

# Predictive UP 48
# sbatch -A your_account -p booster --nodes $NNODES --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=pred-up-scaling-$NNODES experiments/run_gb_script.sh systems/jupiter_container.sh src/main.py  --skip_validation --config_file configs/predictive_config_jupiter.yaml mesh_dims=-1,48,4,2 data_train_subset=$((50 * $NNODES / 96)) training_n_epochs=2 training_log_interval=5

# Inference DP 4
# sbatch -A your_account \
#     -p booster \
#     -t 30:00 \
#     --gres=gpu:4 \
#     --tasks-per-node 4 \
#     -N $NNODES \
#     --job-name=inference-$NNODES \
#     ./experiments/run_gb_script.sh \
#     systems/jupiter_container.sh \
#     src/main_evaluation.py eval \
#         --rollout 1 \
#         --no-avg_pool_store \
#         /path/to/output/booster/inference \
#         mesh_dims=4,-1,4,2 \
#         training_batch_size=2 \
#         training_n_samples=1 \
#         training_log_interval=5 \
#         data_valid_subset=800 \
