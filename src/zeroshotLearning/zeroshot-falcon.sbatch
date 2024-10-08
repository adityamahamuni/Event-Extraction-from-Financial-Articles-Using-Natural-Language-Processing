#!/bin/bash
#
#SBATCH --job-name=zeroshotEvent # Job name for tracking
#SBATCH --partition=falcon     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1          # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA

python3.11 -m pip install sentencepiece
python3.11 -m pip install -r requirements.txt
python3.11 -m pip install transformers --upgrade

python3.11 zeroshotLearning_PromptingStrategy_Optimized.py --model_name "google/flan-t5-xl" --output_file "flanT5_optimizedRun.json"