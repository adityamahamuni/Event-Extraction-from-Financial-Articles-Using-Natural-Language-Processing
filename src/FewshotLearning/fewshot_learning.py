import torch
import json
import logging
import re
import os
import argparse
import random
from enum import Enum
from rank_bm25 import BM25Okapi
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, classification_report
from transformers import (AutoTokenizer, AutoModelForCausalLM, T5Tokenizer,
                          T5ForConditionalGeneration, BertTokenizer,
                          BertForSequenceClassification, BartTokenizer,
                          BartForConditionalGeneration)


# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enum for different event types
class EventType(Enum):
    A = "Acquisition (A)"
    CT = "Clinical Trial (CT)"
    RD = "Regular Dividend (RD)"
    DC = "Dividend Cut (DC)"
    DI = "Dividend Increase (DI)"
    GI = "Guidance Increase (GI)"
    NC = "New Contract (NC)"
    RSS = "Reverse Stock Split (RSS)"
    SD = "Special Dividend (SD)"
    SR = "Stock Repurchase (SR)"
    SS = "Stock Split (SS)"
    O = "Other/None (O)"


class EventDetectionModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.cache_dir = '/dcs/large/u5579267/.huggingface'

        logging.info(f"Initializing model: {model_name}")

        # Load model and tokenizer based on model type
        self._load_model_and_tokenizer()

        # Set device for model (GPU if available, else CPU)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logging.info(f"Model loaded on device: {self.device}")

        # Load SBERT model for sentence embeddings
        self.sbert_model = self.load_sbert_model()

    def _load_model_and_tokenizer(self):
        """Load the appropriate model and tokenizer based on the model name."""

        if "flan-t5" in self.model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name, legacy=True, cache_dir=self.cache_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name, cache_dir=self.cache_dir)
        elif "bart" in self.model_name.lower():
            self.tokenizer = BartTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir)
            self.model = BartForConditionalGeneration.from_pretrained(
                self.model_name, cache_dir=self.cache_dir)
        elif "gpt" in self.model_name.lower() or "llama" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, cache_dir=self.cache_dir)
        elif "bert" in self.model_name.lower():
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels, cache_dir=self.cache_dir)
        else:
            logging.error("Model not supported - running with pipeline")
            raise ValueError("Model not supported")

    def load_sbert_model(self):
        """Load the SBERT model for sentence embeddings."""
        return SentenceTransformer('all-MiniLM-L6-v2')

    def get_device(self):
        """Get the device on which the model is loaded (GPU or CPU)."""
        return self.device


class EventExtraction:
    def __init__(self, model_initializer, fewshot_strategy="random"):
        self.model_name = model_initializer.model_name
        self.tokenizer = model_initializer.tokenizer
        self.model = model_initializer.model
        self.sbert_model = model_initializer.sbert_model
        self.device = model_initializer.get_device()
        self.fewshot_strategy = fewshot_strategy

    def generate_event_information(self, prompt, max_input_length=500, max_new_tokens=50):
        """Generate event information from the input sentence."""

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            inputs['input_ids'], max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        return generated_text

    def get_model_response(self, prompt, max_new_tokens=512):
        """Get response from the model for the given prompt."""

        logging.debug(f"Generating model response for prompt: {prompt}")
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Model response: {response}")
        return response

    def sbert_selection(self, sentence, training_data, k):
        """Select few-shot examples using SBERT model based on similarity."""

        logging.info("Selecting few-shot examples using SBERT")
        embeddings = self.sbert_model.encode(
            [sentence] + [data['sentence'][0] for data in training_data])
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [training_data[i] for i in top_k_indices]

    def bm25_selection(self, sentence, training_data, k):
        """Select few-shot examples using BM25 algorithm."""

        logging.info("Selecting few-shot examples using BM25")
        tokenized_corpus = [word_tokenize(
            data['sentence'][0]) for data in training_data]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = word_tokenize(sentence)
        top_k_indices = bm25.get_top_n(
            tokenized_query, range(len(training_data)), n=k)
        return [training_data[i] for i in top_k_indices]

    def fewshot_selection(self, sentence, training_data, k=5):
        """Select few-shot examples based on the specified strategy."""

        if self.fewshot_strategy == "random":
            return self.load_few_shot_examples()
        elif self.fewshot_strategy == "sbert":
            return self.sbert_selection(sentence, training_data, k)
        elif self.fewshot_strategy == "bm25":
            return self.bm25_selection(sentence, training_data, k)
        else:
            raise ValueError(f"Invalid strategy: {self.fewshot_strategy}")

    def load_few_shot_examples(self, filename="fewshot_examples.txt"):
        """Load few-shot examples from a file."""

        logging.info(f"Loading few-shot examples from: {filename}")
        with open(filename, 'r') as file:
            lines = file.readlines()

        examples = []
        current_example = {"sentence": "", "event": ""}
        for line in lines:
            if line.startswith("Sentence: "):
                if current_example["sentence"]:
                    examples.append(current_example)
                    current_example = {"sentence": "", "event": ""}
                current_example["sentence"] = line.strip().replace(
                    "Sentence: ", "")
            elif line.startswith("Event: "):
                current_example["event"] = line.strip().replace("Event: ", "")

        if current_example["sentence"]:
            examples.append(current_example)

        return examples

    def get_prompt(self, prompt_type, sentence, training_data, few_shot_examples=[]):
        """Generate a prompt for the model based on the prompt type."""
        fewshot_prompt = ""
        if few_shot_examples and self.fewshot_strategy == 'random':
            fewshot_prompt = "\n\n".join(
                [f"Sentence: {ex['sentence']}\nEvent: {ex['event']}" for ex in few_shot_examples])
        elif few_shot_examples:
            fewshot_prompt = "\n\n".join(
                [f"Sentence: {ex}" for ex in few_shot_examples])

        if prompt_type == "schema":
            return self.get_schema_prompt(sentence, fewshot_prompt)
        elif prompt_type == "code":
            return self.get_code_prompt(sentence, fewshot_prompt)
        elif prompt_type == "explanation":
            return self.get_explanation_prompt(sentence, fewshot_prompt)
        elif prompt_type == "pipeline":
            return self.get_pipeline_prompt(sentence, fewshot_prompt)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

    def get_schema_prompt(self, sentence, fewshot_prompt):
        """Generate a schema prompt for event extraction."""

        return f"""
            Extract event information from the following sentence and return the most matching event as event_type

            Event_type:
            - Acquisition (A)
            - Clinical Trial (CT)
            - Regular Dividend (RD)
            - Dividend Cut (DC)
            - Dividend Increase (DI)
            - Guidance Increase (GI)
            - New Contract (NC)
            - Reverse Stock Split (RSS)
            - Special Dividend (SD)
            - Stock Repurchase (SR)
            - Stock Split (SS)
            - Other/None (O)

            Sentence: "{sentence}"

            Examples:
            {fewshot_prompt}

            Output:
        """

    def process_sentence(self, sentence, training_data):
        """Process a single sentence to extract event information."""

        logging.info(f"Processing sentence: {sentence}")
        few_shot_examples = self.fewshot_selection(sentence, training_data)
        prompt = self.get_prompt(
            "schema", sentence, training_data, few_shot_examples)
        logging.debug(f"Prompt: {prompt}")
        response = self.get_model_response(prompt)
        return {"event_type": response}

    def process_dataset(self, dataset_path, samples=5):
        """Process a dataset and extract events from each sentence."""
        logging.info(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, "r") as file:
            data = json.load(file)

        results = []
        for item in data:
            sentence = item["sentence"][0]
            extracted_events = self.process_sentence(sentence, data)
            results.append({
                "sentence": sentence,
                "extracted_events": extracted_events,
                "actual_events": item["events"]
            })
            torch.cuda.empty_cache()
            logging.info(f"Extracted events: {extracted_events}")

        logging.info("Processing Data Complete")
        logging.info("=" * 50)
        return results

    def extract_event_type(self, text):
        """Extract the event type from the model's output."""

        pattern = re.compile(r'(?<=event_type": ")(.*?)(?=")', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1)
        else:
            return "N/A"

    def save_results(self, results, output_file):
        """Save the extracted results to a JSON file."""

        logging.info(f"Saving results to: {output_file}")
        with open(output_file, "w") as file:
            json.dump(results, file, indent=4)

    def evaluate_events(self, results):
        """Evaluate the extracted events against the actual events."""

        logging.info("Evaluating extracted events")

        y_true = []
        y_pred = []

        event_mapping = {
            'Acquisition (A)': 'A',
            'Clinical Trial (CT)': 'CT',
            'Regular Dividend (RD)': 'RD',
            'Dividend Cut (DC)': 'DC',
            'Dividend Increase (DI)': 'DI',
            'Guidance Increase (GI)': 'GI',
            'New Contract (NC)': 'NC',
            'Reverse Stock Split (RSS)': 'RSS',
            'Special Dividend (SD)': 'SD',
            'Stock Repurchase (SR)': 'SR',
            'Stock Split (SS)': 'SS',
            'Other/None (O)': 'O'
        }

        for result in results:
            actual_events = result["actual_events"]
            extracted_event = result["extracted_events"]["event_type"].strip()

            normalized_extracted_event = event_mapping.get(
                extracted_event, 'O')

            for actual in actual_events:
                normalized_actual_event = event_mapping.get(actual, 'O')
                actual_event_enum = next(
                    (e for e in EventType if e.name == normalized_actual_event), EventType.O)
                y_true.append(actual_event_enum.value)

                matched_event_enum = next(
                    (e for e in EventType if e.name == normalized_extracted_event), EventType.O)
                y_pred.append(matched_event_enum.value)

        unique_labels = sorted(set([e.value for e in EventType]))

        # Calculate Exact Match (EM)
        exact_matches = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        em_score = exact_matches / len(y_true)

        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, average=None)

        # Calculate micro and macro averages
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro')
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro')

        logging.info(
            f"Evaluation metrics - Exact Match (EM): {em_score}, Micro Precision: {micro_precision}, Micro Recall: {micro_recall}, Micro F1: {micro_f1}, Macro Precision: {macro_precision}, Macro Recall: {macro_recall}, Macro F1: {macro_f1}")
        logging.info("=" * 50)

        return {
            "exact_match": em_score,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1
        }


# Function to analyze the distribution of predicted events
def analyze_predictions(results):
    class_counts = {}
    for result in results:
        predicted = result["extracted_events"]["event_type"]
        if predicted in class_counts:
            class_counts[predicted] += 1
        else:
            class_counts[predicted] = 1

    logging.info(f"Class distribution in predictions: {class_counts}")
    return class_counts



def main():
    parser = argparse.ArgumentParser(description="Event Extraction Script")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-xl",
                        help="Model name to use for event extraction")
    parser.add_argument("--dataset_path", type=str, default="EDT_dataset/Event_detection/train.json",
                        help="Path to the dataset file")
    parser.add_argument("--output_file", type=str, default="extracted_events.json",
                        help="Name of the output file to save results")
    parser.add_argument("--fewshot_strategy", type=str, default='random',
                        help="Few-shot strategy to use: random, sbert, bm25")

    args = parser.parse_args()

    dataset_path = os.path.join(os.getcwd(), args.dataset_path)
    model_name = args.model_name

    logging.info("Starting event extraction process")
    model_initializer = EventDetectionModel(model_name)
    event_extractor = EventExtraction(
        model_initializer, fewshot_strategy=args.fewshot_strategy)

    logging.info(f"Few-shot strategy: {args.fewshot_strategy}")
    logging.info(f"Model Name: {model_name}")
    logging.info("Few-shot Event Extraction Process Started")

    prompt_types = ['schema']

    for prompt_type in prompt_types:
        logging.info(f"Prompt Type: {prompt_type}")

        output_file = args.output_file
        output_file = output_file.replace(".json", f"_{prompt_type}.json")
        output_path = os.path.join(os.getcwd(), output_file)

        # Process the dataset to extract events
        results = event_extractor.process_dataset(dataset_path)

        # Analyze the distribution of predictions
        prediction_distribution = analyze_predictions(results)
        logging.info(f"Prediction Distribution: {prediction_distribution}")

        # Save results to file
        event_extractor.save_results(results, output_path)

        logging.info(
            f"Extraction completed for prompt type: {prompt_type} - Results saved to {output_path} - Evaluating results")

        # Evaluate the extracted events
        evaluation_metrics = event_extractor.evaluate_events(results)
        logging.info(f"Evaluation Metrics {prompt_type}: {evaluation_metrics}")
        logging.info("=" * 50)

    logging.info("Few-shot Event Extraction Process Completed")


if __name__ == "__main__":
    main()
