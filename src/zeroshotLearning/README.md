# Zero-Shot Learning Event Extraction

The zeroshot learning demonstrates how to perform event extraction using zero-shot learning with various pre-trained language models. The extraction process involves identifying and categorizing events from textual data, such as financial news or company announcements, without any task-specific fine-tuning.

## Overview

The script leverages pre-trained models from Hugging Face to perform event extraction in a zero-shot manner. It supports multiple models such as FLAN-T5, BART, GPT. The script processes a dataset of sentences, generates prompts for the models, and extracts event types based on predefined categories.


## Usage

### Arguments

- `--model_name`: The name of the pre-trained model to use for event extraction (e.g., `google/flan-t5-xl`).
- `--dataset_path`: Path to the dataset file (JSON format).
- `--output_file`: Name of the output file where results will be saved (default: `extracted_events.json`).

### Running the Script

To run the script, use the following command:

```bash
python3.11 zeroshotPrompting_Metrics.py --model_name "google/flan-t5-xl" --dataset_path "/dcs/large/u5579267/EventExtraction/EDT_dataset/Event_detection/train.json" --output_file "output.json"
```

## Models Supported

The script supports the following models:

- **FLAN-T5**: Google's fine-tuned version of T5 for tasks requiring instruction following.
- **BART**: Facebook's BART model for sequence-to-sequence tasks.
- **GPT**: Models from the GPT series, such as GPT-2.

## Prompt Types

The script supports different prompt types, which define how the input sentences are presented to the model:

- **Schema**: Presents a list of possible event types for the model to choose from.
- **Code**: Mimics a coding-style prompt where events are defined as variables.
- **Explanation**: Provides detailed descriptions of each event type.
- **Pipeline**: Uses a staged approach, first extracting the event type and then explaining it.

## Evaluation Metrics

The script evaluates the extracted events against the actual events in the dataset using the following metrics:

- **Exact Match (EM)**: The proportion of sentences where the predicted event matches the actual event exactly.
- **F1 Score**: A weighted average of precision and recall.

## Results

The results of the extraction process, including the extracted events and evaluation metrics, are saved in a JSON file specified by the `--output_file` argument. 

Example result:

```json
{
    "sentence": "Company ABC announces a new contract with XYZ Corp.",
    "extracted_events": "New Contract (NC)",
    "actual_events": ["New Contract (NC)"]
}
```
