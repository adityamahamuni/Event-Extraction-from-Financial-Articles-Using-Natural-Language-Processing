#!/bin/bash
#
#SBATCH --job-name=zeroshotEvent-cpu # Job name for tracking
#SBATCH --partition=parrot  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=24     # Number of CPU threads used by your job
#SBATCH --mem=120000           # 120GB RAM
#SBATCH --exclusive=mcs        # Exclusive mode, only this job will run
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##

python3.11 -m pip install sentencepiece
python3.11 -m pip install -r requirements.txt
python3.11 -m pip install transformers --upgrade

python3.11 zeroshotPrompting.py --model_name "google/flan-t5-xl" --output_file "flanT5_classificationReportOutput.json"
