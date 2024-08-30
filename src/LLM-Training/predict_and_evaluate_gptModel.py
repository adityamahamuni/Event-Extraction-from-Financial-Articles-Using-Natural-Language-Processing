import json
import os
import logging
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
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
        self.data = self.load_json()
        self.label_mapping = self.create_label_mapping()

    def load_json(self):
        """Load the dataset from a JSON file."""

        logger.info("Loading JSON data.")
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def create_label_mapping(self):
        """Create a mapping from event labels to numerical indices."""

        logger.info("Creating label mapping.")
        unique_labels = {entry['events'][0] for entry in self.data}
        label_mapping = {label: idx for idx,
                         label in enumerate(sorted(unique_labels))}
        logger.info(f"Label mapping: {label_mapping}")
        return label_mapping

    def process_data(self):
        """Process the data into a format suitable for model evaluation."""

        logger.info("Processing data for prediction.")
        sentences = [entry['sentence'][0] for entry in self.data]
        labels = [self.label_mapping[entry['events'][0]]
                  for entry in self.data]
        return Dataset.from_dict({"text": sentences, "label": labels})

    def get_dataset(self):
        """Return the processed dataset as a HuggingFace Dataset object."""
        logger.info("Creating HuggingFace Dataset object.")
        return self.process_data()


class ModelEvaluator:
    def __init__(self, model_dir, dataset, eval_batch_size=1, use_cpu=False, seed=42):
        """Initialize the ModelEvaluator with necessary parameters."""

        self.model_dir = model_dir
        self.dataset = dataset
        self.use_cpu = use_cpu
        self.eval_batch_size = eval_batch_size

        # Set the seed for reproducibility
        set_seed(seed)

        logger.info("Loading the model and tokenizer.")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not use_cpu else "cpu")

        # Load tokenizer and model, with fallback to saved directories if not found
        self.tokenizer = self.load_tokenizer(model_dir)
        self.model = self.load_model(model_dir)

    def load_tokenizer(self, model_dir):
        """Load the tokenizer from the model directory, with fallback."""

        try:
            return AutoTokenizer.from_pretrained(model_dir)
        except OSError:
            logger.warning(
                f"Tokenizer not found in {model_dir}. Falling back to the original model directory.")
            original_model_dir = os.path.join(model_dir, "saved_tokenizer")
            return AutoTokenizer.from_pretrained(original_model_dir)

    def load_model(self, model_dir):
        """Load the model from the model directory, with fallback."""

        try:
            return AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        except OSError:
            logger.warning(
                f"Model not found in {model_dir}. Falling back to the original model directory.")
            original_model_dir = os.path.join(model_dir, "saved_model")
            return AutoModelForSequenceClassification.from_pretrained(original_model_dir).to(self.device)

    def tokenize_function(self, examples):
        """Tokenize the input text."""
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def preprocess_dataset(self):
        """Tokenize the dataset."""
        logger.info("Tokenizing the dataset.")
        return self.dataset.map(self.tokenize_function, batched=True)

    def compute_metrics(self, pred):
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
        tokenized_dataset = self.preprocess_dataset()

        # Using smaller batch size for evaluation to reduce memory usage
        training_args = TrainingArguments(
            per_device_eval_batch_size=self.eval_batch_size,
            output_dir=self.model_dir,
            no_cuda=self.use_cpu,
            fp16=not self.use_cpu,  # Use FP16 if using GPU
            seed=42
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics
        )

        logger.info("Making predictions.")
        try:
            predictions = trainer.predict(tokenized_dataset)
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA out of memory during evaluation. Consider lowering the evaluation batch size.")
            raise

        logger.info("Computing metrics.")
        eval_results = self.compute_metrics(predictions)
        logger.info(f"Evaluation Results: {eval_results}")

        self.save_results(eval_results)

    def save_results(self, eval_results):
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
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Batch size for evaluation. Lower this if you encounter CUDA OOM errors.")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Use CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Process the data
    processor = DatasetProcessor(args.dataset_path)
    dataset = processor.get_dataset()

    # Evaluate the model
    evaluator = ModelEvaluator(
        model_dir=args.model_dir,
        dataset=dataset,
        eval_batch_size=args.eval_batch_size,
        use_cpu=args.use_cpu,
        seed=args.seed
    )
    evaluator.evaluate()
