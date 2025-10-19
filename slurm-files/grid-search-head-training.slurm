#! /bin/sh

#SBATCH --account=gpu-students
#SBATCH --job-name=hydra_sweep_head_training
#SBATCH --chdir=/home/joberant/NLP_2425b/tslilshani/historical-text-dating
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50000
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-user=tslilshani@mail.tau.ac.il
#SBATCH --mail-type=FAIL

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# activate conda environment
source /home/joberant/NLP_2425b/tslilshani/anaconda3/etc/profile.d/conda.sh
conda activate nlp_2025

python -m src.main -m training=head-training hydra/launcher=submitit_slurm
