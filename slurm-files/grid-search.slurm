#! /bin/sh

#SBATCH --job-name=hydra_sweep
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50000
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

python -m src.main -m training=head-training hydra/launcher=submitit_slurm