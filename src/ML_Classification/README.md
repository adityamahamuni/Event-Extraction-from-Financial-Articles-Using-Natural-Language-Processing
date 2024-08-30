### README: Event Classification with LSTM and ML Algorithms

This folder contains two Python scripts for text classification tasks, specifically event classification from text data. The scripts employ different approaches: one uses a deep learning model (BiLSTM with Attention), and the other leverages traditional machine learning algorithms.


## Prerequisites

Ensure you have Python 3.8 or higher installed on your machine. Both scripts require specific Python libraries, which are listed in the `requirements.txt` file.

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


### Overview

The script (`eventClassification_LSTM.py`) implements a LSTM (Bidirectional LSTM) model for event classification. The model is trained to classify text sequences into predefined event categories.

### Command-Line Arguments

- `--dataset_path`: The path to the dataset file (in JSON format).
- `--model_name`: The name used for saving the model, tokenizer, and label encoder files.

### Example Command

```bash
python3.11 eventClassification_LSTM.py --dataset_path "/dcs/large/u5579267/EventExtraction/EDT_dataset/Event_detection/train.json" --model_name "lstm-model"
```

### Output

- **Trained Model**
- **Tokenizer and Label Encoder**
- **Evaluation Metrics:** Accuracy, precision, recall, and F1 score are displayed after evaluation.



The script (`eventClassification.py`) uses traditional machine learning algorithms to classify events based on text data. Several algorithms are trained, and the best model is selected based on cross-validation performance.

### Command-Line Arguments

- `--dataset_path`: The path to the dataset file (in JSON format).
- `--model_name`: The name used for saving the model.
- 
### Example Command

```bash
python3.11 eventClassification.py --dataset_path "/dcs/large/u5579267/EventExtraction/EDT_dataset/Event_detection/train.json"
```

### ML Model Details

- **Algorithms Used:**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - XGBoost

- **Model Selection:** The best model is selected based on grid search with cross-validation.
