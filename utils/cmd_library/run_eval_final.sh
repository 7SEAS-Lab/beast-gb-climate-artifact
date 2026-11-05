#!/usr/bin/env bash
# JUPITER
#
subsampling=${1:-12}
export ROLLOUT=${2:-15}
export BEAST_DS_HOURLY_SUBSAMPLING=${subsampling}

samples=192
nodes=$(( samples * 4 ))

sbatch -A your_account \
    -p booster \
    -t 3:00:00 \
    --gres=gpu:4 \
    --tasks-per-node 4 \
    -N $nodes \
    --job-name=eval-2022-N${nodes}-r${ROLLOUT}_subs${subsampling}-gss${samples} \
    --output=logs/eval/%x_%j.out \
    --error=logs/eval/%x_%j.err \
    ./experiments/run_gb_script.sh \
    systems/jupiter_container.sh \
    src/main_evaluation.py eval \
        --rollout $ROLLOUT \
        --no-avg_pool_store \
        /path/to/output/booster/inference \
        mesh_dims=4,${samples},2,2 \
        training_batch_size=4 \
        training_n_samples=1 \
        training_log_interval=10 \
        training_checkpoint_interval=100 \
