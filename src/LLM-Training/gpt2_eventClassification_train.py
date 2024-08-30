import json
import os
import time
import logging
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._load_json()
        self.label_mapping = self._create_label_mapping()

    def _load_json(self):
        """Load data from the specified JSON file."""

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
        """Process the data into sentences and corresponding labels."""

        logger.info("Processing data for training.")
        sentences = [entry['sentence'][0] for entry in self.data]
        labels = [self.label_mapping[entry['events'][0]]
                  for entry in self.data]
        return Dataset.from_dict({"text": sentences, "label": labels})

    def get_dataset(self):
        """Return the processed dataset as a HuggingFace Dataset object."""

        logger.info("Creating HuggingFace Dataset object.")
        return self._process_data()


class ModelTrainer:
    def __init__(self, model_name, dataset, label_mapping, test_size=0.2, batch_size=3, epochs=3, output_dir='.', quantized=False):
        self.model_name = model_name
        self.dataset = dataset
        self.label_mapping = label_mapping
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir

        # Load tokenizer and handle padding token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._set_padding_token()

        # Determine the number of labels based on the label mapping
        num_labels = len(self.label_mapping)

        # Load the model, optionally as a quantized model
        self.model = self._load_model(num_labels, quantized)

        # Resize token embeddings after adding a new pad token
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _set_padding_token(self):
        """Ensure the tokenizer has a padding token."""

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens(
                {'pad_token': self.tokenizer.eos_token})
            logger.info(
                f"Padding token set to EOS token: {self.tokenizer.pad_token}")

    def _load_model(self, num_labels, quantized):
        """Load the model with the specified number of labels."""

        if quantized:
            logger.info("Loading quantized model.")
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels, torch_dtype=torch.float16)
        else:
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)

    def _tokenize_function(self, examples):
        """Tokenize the input text."""
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def _preprocess_dataset(self):
        """Tokenize the dataset and split it into training and test sets."""

        logger.info("Tokenizing the dataset.")
        tokenized_datasets = self.dataset.map(
            self._tokenize_function, batched=True)
        return tokenized_datasets.train_test_split(test_size=self.test_size)

    def _compute_metrics(self, pred):
        """Compute metrics for model evaluation."""

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

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

    def train(self):
        """Train the model and evaluate its performance."""

        logger.info("Starting the training process.")
        dataset_dict = self._preprocess_dataset()
        train_dataset = dataset_dict['train']
        test_dataset = dataset_dict['test']

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_steps=10,
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self._compute_metrics
        )

        logger.info("Training the model.")
        start_train_time = time.time()
        trainer.train()
        end_train_time = time.time()
        logger.info(
            f"Training completed in {end_train_time - start_train_time:.2f} seconds.")

        logger.info("Evaluating the model.")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation Results: {eval_results}")

        self._save_results(eval_results)
        self._save_model()

        logger.info("Training and evaluation completed successfully.")

    def _save_results(self, eval_results):
        """Save evaluation results to a JSON file."""

        results_file = f'results-{self.model_name.split("/")[-1]}.json'
        results_path = os.path.join(self.output_dir, results_file)
        logger.info(f"Saving evaluation results to {results_path}.")
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=4)

    def _save_model(self):
        """Save the trained model and tokenizer."""

        logger.info("Saving the model and tokenizer.")
        model_save_path = os.path.join(self.output_dir, "saved_model")
        tokenizer_save_path = os.path.join(self.output_dir, "saved_tokenizer")

        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(tokenizer_save_path)

        logger.info(f"Model saved to {model_save_path}")
        logger.info(f"Tokenizer saved to {tokenizer_save_path}")

    def load_model(self):
        """Load the model and tokenizer from the saved directory."""

        logger.info("Loading the model and tokenizer.")
        model_load_path = os.path.join(self.output_dir, "saved_model")
        tokenizer_load_path = os.path.join(self.output_dir, "saved_tokenizer")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)

        logger.info(f"Model loaded from {model_load_path}")
        logger.info(f"Tokenizer loaded from {tokenizer_load_path}")


if __name__ == "__main__":
    start_program_time = time.time()

    parser = argparse.ArgumentParser(
        description="Train a model for event extraction from corporate announcements.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model to use (e.g., google/flan-t5-small).")
    parser.add_argument("--batch_size", type=int, default=3,
                        help="Batch size for training.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset file (JSON format).")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs for training.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save the trained model and outputs.")
    parser.add_argument("--quantized", action='store_true',
                        help="Use quantized model (for large models like llama3).")

    args = parser.parse_args()

    # Process the dataset
    processor = DatasetProcessor(args.dataset_path)
    dataset = processor.get_dataset()

    # Train the model
    trainer = ModelTrainer(
        model_name=args.model_name,
        dataset=dataset,
        label_mapping=processor.label_mapping,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        quantized=args.quantized
    )
    trainer.train()

    end_program_time = time.time()
    logger.info(
        f"Total program runtime: {end_program_time - start_program_time:.2f} seconds.")
