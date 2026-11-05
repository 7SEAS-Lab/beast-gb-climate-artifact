#!/usr/bin/env bash
# Cyclone
# sbatch -A your_account \
#     -p booster \
#     -t 30:00 \
#     --gres=gpu:4 \
#     --tasks-per-node 4 \
#     -N 96 \
#     ./experiments/run_gb_script.sh \
#     systems/jupiter_container.sh \
#     src/main_evaluation.py ewb \
#         --avg_pool_store \
#         --lead_times=0,-48 \
#         /path/to/output/booster/inference \
#         mesh_dims=1,-1,2,2 \
#         training_n_samples=1 \
#         data_ewb_case_ids=events_tropical_cyclone_256

# Heatwave
# sbatch -A your_account \
#     -p booster \
#     -t 30:00 \
#     --gres=gpu:4 \
#     --tasks-per-node 4 \
#     -N 256 \
#     ./experiments/run_gb_script.sh \
#     systems/jupiter_container.sh \
#     src/main_evaluation.py ewb \
#         --avg_pool_store \
#         --lead_times=240:-180:-6 \
#         /path/to/output/booster/inference \
#         mesh_dims=1,-1,2,2 \
#         training_n_samples=1 \
#         data_ewb_case_ids=events_heat_wave_20
#
# Freeze
sbatch -A your_account \
    -p booster \
    -t 30:00 \
    --gres=gpu:4 \
    --tasks-per-node 4 \
    -N 256 \
    ./experiments/run_gb_script.sh \
    systems/jupiter_container.sh \
    src/main_evaluation.py ewb \
        --avg_pool_store \
        --lead_times=240:-228:-6 \
        /path/to/output/booster/inference \
        mesh_dims=1,-1,2,2 \
        training_n_samples=1 \
        data_ewb_case_ids=events_freeze_91
