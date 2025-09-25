#! /bin/sh

#SBATCH --job-name=historical-text-dating
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2425b/yonatanl/NLP_2025/historical-text-dating/outputs/historical-text-dating.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2425b/yonatanl/NLP_2025/historical-text-dating/outputs/historical-text-dating.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

python src/main.py

