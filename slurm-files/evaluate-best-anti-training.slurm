#!/bin/sh

#SBATCH --job-name=evaluate_best
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50000
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1


python -m src.main -m training=head-training +is_final_run=true +tracker.job_type=final_evaluation hydra.sweeper.params={} training.seed=45,49,128 training.batch_size=32 training.learning_rate=5e-05 training.max_epochs=7 hydra/launcher=submitit_slurm model.model_name_from_wandb=google-bert-bert-base-multilingual-cased-nwgrl4xi model.freeze_encoder=false