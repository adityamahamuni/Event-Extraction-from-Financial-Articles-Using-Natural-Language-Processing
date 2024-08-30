# Few-Shot Event Extraction

This project focuses on event extraction using few-shot learning techniques. The code supports various strategies for selecting few-shot examples and utilizes different models like `Flan-T5` for event extraction.

## Prerequisites

Before running the program, ensure that you have the following:

- Python 3.11 installed.
- Necessary Python packages as specified in the `requirements.txt` file.
- CUDA installed (for GPU usage).

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <repo_name>
   cd <folder>
   ```

2. **Install Dependencies**:

   If you're running the script on a local machine, you can manually install the dependencies using:

   ```bash
   python3.11 -m pip install -r requirements.txt
   python3.11 -m pip install transformers --upgrade
   python3.11 -m pip install accelerate -U
   python3.11 -m pip install sentencepiece
   ```

## Running the Program

### Using `sbatch` (HPC Cluster)

For running the program on a High-Performance Computing (HPC) cluster using `sbatch`, you need to prepare an `sbatch` script like the one below:

#### Submitting the Job

To submit the job to the HPC cluster, use the following command:

```bash
sbatch your_sbatch_script.sh
```

### Running Locally with Python

If you want to run the program on your local machine, follow these steps:

1. **Ensure that all dependencies are installed**:
   
   ```bash
   python3.11 -m pip install -r requirements.txt
   python3.11 -m pip install transformers --upgrade
   python3.11 -m pip install accelerate -U
   python3.11 -m pip install sentencepiece
   ```

2. **Run the Python Script**:

   You can run the event extraction script with the following command:

   ```bash
   python3.11 ./fewshot_learning.py --model_name "google/flan-t5-xl" --fewshot_strategy "bm25" --dataset_path "/path/to/dataset/train.json" --output_file "fewshot_bm25_full.json"
   ```


### Command-Line Arguments

1. **`--model_name`**
   - **Description:** Specifies the pre-trained model to be used for event extraction.
   - **Default Value:** `"google/flan-t5-xl"`
   - **Example Usage:**
     ```bash
     --model_name "bert-base-uncased"
     ```
   - **Explanation:** This argument determines which transformer model (from Hugging Face) will be loaded and used. It can be any model that is supported by the `transformers` library, such as `bert-base-uncased`, `google/flan-t5-xl`, `bart-large`, etc.

2. **`--dataset_path`**
   - **Description:** Specifies the path to the dataset file that contains sentences and their corresponding events for training or evaluation.
   - **Default Value:** `"EDT_dataset/Event_detection/train.json"`
   - **Example Usage:**
     ```bash
     --dataset_path "/path/to/your/dataset/train.json"
     ```
   - **Explanation:** This argument points to the JSON file containing the dataset. The dataset is expected to have a structure where each entry includes a sentence and associated event(s). The script processes this dataset to perform event extraction.

3. **`--output_file`**
   - **Description:** Defines the name of the output file where the results of the event extraction will be saved.
   - **Default Value:** `"extracted_events.json"`
   - **Example Usage:**
     ```bash
     --output_file "results/output_events.json"
     ```
   - **Explanation:** After processing the dataset, the script will save the results (including extracted events) in this file. The file will be in JSON format and contain the input sentences, predicted events, and actual events for each entry.

4. **`--fewshot_strategy`**
   - **Description:** Determines the strategy used for selecting few-shot examples to guide the model's predictions.
   - **Default Value:** `"random"`
   - **Possible Values:** `"random"`, `"sbert"`, `"bm25"`
   - **Example Usage:**
     ```bash
     --fewshot_strategy "bm25"
     ```
   - **Explanation:**
     - `"random"`: Randomly selects few-shot examples from the dataset.
     - `"sbert"`: Uses Sentence-BERT (SBERT) to select the most similar few-shot examples based on sentence embeddings.
     - `"bm25"`: Uses the BM25 algorithm to select the most relevant few-shot examples based on token similarity.

### Example Command

```bash
python3.11 ./fewshot_learning.py --model_name "google/flan-t5-xl" --fewshot_strategy "bm25" --dataset_path "/dcs/large/u5579267/EventExtraction/EDT_dataset/Event_detection/train.json" --output_file "fewshot_bm25_full.json"
```
