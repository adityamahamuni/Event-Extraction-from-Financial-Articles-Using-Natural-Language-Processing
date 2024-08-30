# Large Language Model (LLM) Training, Evaluation, and Validation

This folder contains scripts for training, evaluating, and validating large language models (LLMs) using the Hugging Face `transformers` library. The scripts are designed to handle sequence classification tasks, such as event extraction from text data.

## Prerequisites

Before starting, ensure you have the necessary dependencies installed. The `requirements.txt` file contains all the required packages. You will need Python 3.8 or higher.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repo_name>
   cd <folder>
   ```

2. **Install Dependencies:**

   Install the required Python packages using the following command:

   ```bash
   pip install -r requirements.txt
   ```


## Model Training


The flanT5_eventClassification_train.py and the gpt2_eventClassification_train.py scripts are used to train a large language model on the event dataset.

### Example Command

```bash
python3.11 flanT5_eventClassification_train.py --model_name "google/flan-t5-base" --model_type "flan-t5" --batch_size 3 --eval_batch_size 3 --epochs 5 --dataset_path "/dcs/large/u5579267/EventExtraction/EDT_dataset/Event_detection/train.json" --output_dir .
```

### Training Output

The script will save the following:
- **Trained Model:** The fine-tuned model will be saved in the specified `output_dir`.
- **Training Logs:** Logs of the training process, including loss, accuracy, and other metrics.
- **Evaluation Metrics:** Metrics from evaluating the model on the validation split of the dataset.

## Model Evaluation


The predict_and_evaluate_flanModel.py and predict_and_evaluate_gptModel.py scripts are used to evaluate the performance of a trained model on a separate test dataset. This script will output various metrics, including precision, recall, F1 scores, and accuracy.

### Example Command

```bash
python3.11 ./predict_and_evaluate_flanModel.py --model_dir "/dcs/large/u5579267/EventExtraction/src/LLM-Training/final_model" --dataset_path "/dcs/large/u5579267/EventExtraction/EDT_dataset/Event_detection/train.json" --eval_batch_size 3 --use_cpu
```

### Evaluation Output

The script will save the evaluation results in a JSON file within the `model_dir`, providing detailed metrics on the model's performance.
