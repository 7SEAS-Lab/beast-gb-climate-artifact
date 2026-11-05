#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name='gb_data_conversion'
#SBATCH --output='data_conversion_final.out'
#SBATCH --constraint=LSDF

source /path/to/venv/bin/activate

python -u calculate_normalization.py --data_path "/path/to/data/era5_preprocessed.zarr" --output_path "/path/to/output/norms.nc" --checkpoint_path "/path/to/output/checkpoint1"
