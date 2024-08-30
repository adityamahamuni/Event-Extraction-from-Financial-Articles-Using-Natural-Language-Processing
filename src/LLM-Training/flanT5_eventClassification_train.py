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
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(self, json_path):
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
        unique_labels = set(entry['events'][0] for entry in self.data)
        label_mapping = {label: idx for idx,
                         label in enumerate(sorted(unique_labels))}
        logger.info(f"Label mapping: {label_mapping}")
        return label_mapping

    def _process_data(self):
        """Process the raw data into a format suitable for training."""

        logger.info("Processing data for training.")
        sentences = [entry['sentence'][0] for entry in self.data]
        labels = [self.label_mapping[entry['events'][0]]
                  for entry in self.data]
        return Dataset.from_dict({"text": sentences, "label": labels})

    def get_dataset(self):
        """Return the processed dataset as a Hugging Face Dataset object."""
        logger.info("Creating HuggingFace Dataset object.")
        return self._process_data()


class CustomModelForSequenceClassification(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """Define the forward pass for the model."""

        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits


class ModelTrainer:
    def __init__(self, model_name, dataset, test_size=0.2, batch_size=3, epochs=3, output_dir='.', quantized=False):
        self.model_name = model_name
        self.dataset = dataset
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir
        self.quantized = quantized

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._ensure_padding_token()

        num_labels = len(set(dataset['label']))
        self.model = self._load_model(num_labels)

    def _ensure_padding_token(self):
        """Ensure the tokenizer has a padding token."""

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, num_labels):
        """Load the model, optionally as a quantized model."""

        logger.info("Loading model.")
        if self.quantized:
            return CustomModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                torch_dtype=torch.float16
            )
        else:
            return CustomModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )

    def tokenize_function(self, examples):
        """Tokenize the text data."""

        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def preprocess_dataset(self):
        """Tokenize and split the dataset into training and testing sets."""

        logger.info("Tokenizing the dataset.")
        tokenized_datasets = self.dataset.map(
            self.tokenize_function, batched=True)
        return tokenized_datasets.train_test_split(test_size=self.test_size)

    def compute_metrics(self, pred):
        """Compute evaluation metrics: precision, recall, F1 scores, and accuracy."""

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
            'Accuracy': accuracy
        }

    def train(self):
        """Train the model and evaluate its performance."""

        logger.info("Starting the training process.")
        dataset_dict = self.preprocess_dataset()
        train_dataset = dataset_dict['train']
        test_dataset = dataset_dict['test']

        set_seed(42)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,  # L2 Regularization (Weight Decay)
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,  # Load best model at the end of training
            seed=42,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,

            # Early stopping after 3 epochs without improvement
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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

        self._save_evaluation_results(eval_results)
        self._save_model_and_tokenizer()

        logger.info("Training and evaluation completed successfully.")

    def _save_evaluation_results(self, eval_results):
        """Save the evaluation results to a JSON file."""

        results_file = f'results-{self.model_name.split("/")[-1]}.json'
        results_path = os.path.join(self.output_dir, results_file)
        logger.info(f"Saving evaluation results to {results_path}.")
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=4)

    def _save_model_and_tokenizer(self):
        """Save the trained model and tokenizer."""

        logger.info("Saving the model and tokenizer.")
        save_directory = os.path.join(self.output_dir, "final_model")
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Model and tokenizer saved to {save_directory}.")


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
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        quantized=args.quantized
    )
    trainer.train()

    end_program_time = time.time()
    logger.info(
        f"Total program runtime: {end_program_time - start_program_time:.2f} seconds.")
