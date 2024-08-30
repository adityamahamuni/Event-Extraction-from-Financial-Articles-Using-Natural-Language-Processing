#!/bin/bash
#
#SBATCH --job-name=fewshotEvent # Job name for tracking
#SBATCH --partition=gecko,falcon,eagle     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48     # Number of CPU threads used by your job
#SBATCH --gres=gpu:3           # Number of GPUs to use
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.11 -m pip install sentencepiece
python3.11 -m pip install -r /dcs/large/u5579267/EventExtraction/requirements.txt
python3.11 -m pip install transformers --upgrade
python3.11 -m pip install accelerate -U

python3.11 ./fewshot_learning.py --model_name "google/flan-t5-xl" --fewshot_strategy "bm25" --dataset_path "/dcs/large/u5579267/EventExtraction/EDT_dataset/Event_detection/train.json" --output_file "fewshot_bm25_full.json" 