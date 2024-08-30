import json
import os
import logging
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    T5ForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(self, json_path):
        """Initialize the DatasetProcessor with the path to the JSON data file."""

        self.json_path = json_path
        self.data = self._load_json()
        self.label_mapping = self._create_label_mapping()

    def _load_json(self):
        """Load the dataset from a JSON file."""

        logger.info("Loading JSON data.")
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def _create_label_mapping(self):
        """Create a mapping from event labels to numerical indices."""

        logger.info("Creating label mapping.")
        unique_labels = {entry['events'][0] for entry in self.data}
        label_mapping = {label: idx for idx,
                         label in enumerate(sorted(unique_labels))}
        logger.info(f"Label mapping: {label_mapping}")
        return label_mapping

    def _process_data(self):
        """Process the data into a format suitable for model evaluation."""

        logger.info("Processing data for evaluation.")
        sentences = [entry['sentence'][0] for entry in self.data]
        labels = [self.label_mapping[entry['events'][0]]
                  for entry in self.data]
        return Dataset.from_dict({"text": sentences, "label": labels})

    def get_dataset(self):
        """Return the processed dataset as a HuggingFace Dataset object."""
        logger.info("Creating HuggingFace Dataset object.")
        return self._process_data()


class ModelEvaluator:
    def __init__(self, model_dir, dataset, num_labels, eval_batch_size=8, use_cpu=False, seed=42):
        """Initialize the ModelEvaluator with the necessary parameters."""
        self.model_dir = model_dir
        self.dataset = dataset
        self.use_cpu = use_cpu

        set_seed(seed)

        logger.info("Loading the model and tokenizer.")
        self.tokenizer = self._load_tokenizer(model_dir)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
        self.model = T5ForSequenceClassification.from_pretrained(
            model_dir, num_labels=num_labels).to(self.device)

        self.eval_batch_size = eval_batch_size

    def _load_tokenizer(self, model_dir):
        """Load the tokenizer from the model directory or fallback to the original directory."""

        try:
            return AutoTokenizer.from_pretrained(model_dir)
        except OSError:
            logger.warning(
                f"Tokenizer not found in {model_dir}. Falling back to the original model directory.")
            original_model_dir = os.path.join(os.getcwd(), model_dir)
            return AutoTokenizer.from_pretrained(original_model_dir)

    def _tokenize_function(self, examples):
        """Tokenize the input text."""
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def _preprocess_dataset(self):
        """Tokenize the dataset."""

        logger.info("Tokenizing the dataset.")
        return self.dataset.map(self._tokenize_function, batched=True)

    def _compute_metrics(self, pred):
        """Compute evaluation metrics such as precision, recall, F1 scores, and accuracy."""

        logits = pred.predictions[0] if isinstance(
            pred.predictions, tuple) else pred.predictions
        preds = logits.argmax(-1)
        labels = pred.label_ids

        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1_macro = f1_score(labels, preds, average='macro')
        f1_micro = f1_score(labels, preds, average='micro')
        accuracy = accuracy_score(labels, preds)

        return {
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'accuracy': accuracy
        }

    def evaluate(self):
        """Evaluate the model on the provided dataset."""

        logger.info("Starting the evaluation process.")
        tokenized_dataset = self._preprocess_dataset()

        training_args = TrainingArguments(
            per_device_eval_batch_size=self.eval_batch_size,
            output_dir=self.model_dir,
            no_cuda=self.use_cpu,
            fp16=True if not self.use_cpu else False,
            seed=42
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self._compute_metrics
        )

        logger.info("Making predictions.")
        predictions = trainer.predict(tokenized_dataset)

        logger.info("Computing metrics.")
        eval_results = self._compute_metrics(predictions)
        logger.info(f"Evaluation Results: {eval_results}")

        self._save_results(eval_results)

        logger.info("Evaluation completed successfully.")

    def _save_results(self, eval_results):
        """Save the evaluation results to a JSON file."""

        results_file = 'evaluation_results.json'
        results_path = os.path.join(self.model_dir, results_file)
        logger.info(f"Saving evaluation results to {results_path}.")
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a test dataset.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory where the trained model and tokenizer are saved.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the test dataset file (JSON format).")
    parser.add_argument("--eval_batch_size", type=int,
                        default=8, help="Batch size for evaluation.")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Use CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Process the JSON data
    processor = DatasetProcessor(args.dataset_path)
    dataset = processor.get_dataset()

    # Evaluate the model
    num_labels = len(processor.label_mapping)
    evaluator = ModelEvaluator(
        model_dir=args.model_dir,
        dataset=dataset,
        num_labels=num_labels,
        eval_batch_size=args.eval_batch_size,
        use_cpu=args.use_cpu,
        seed=args.seed
    )
    evaluator.evaluate()
