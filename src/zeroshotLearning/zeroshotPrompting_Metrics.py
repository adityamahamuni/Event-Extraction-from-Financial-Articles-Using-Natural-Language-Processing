import os
import json
import torch
import re
import logging
import argparse
from enum import Enum

from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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


class EventExtraction:
    def __init__(self, model_name):
        logging.info(f"Using {model_name} for event extraction")

        if "flan-t5" in model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name, legacy=True)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif "bart" in model_name.lower():
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name)
        elif "gpt" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif "llama" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            print("Model not supported - running with pipeline")

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.zero_shot_classifier = pipeline(
            "zero-shot-classification", model=model_name, device=self.device)

        logging.info("Initialized EventExtraction model")

    @staticmethod
    def extract_company_name(sentence):
        """Extracts the company name from the sentence based on common patterns."""

        pattern = r"\b[A-Z][a-zA-Z]*\s(?:Inc|Corp|Ltd|LLC|Group|Company|PLC|Corporation|Incorporated|N\.A\.)\b"
        match = re.search(pattern, sentence)
        if match:
            return match.group()
        return "Unknown"

    def extract_events_zero_shot(self, sentence, event_types):
        """
        Performs zero-shot event extraction and company name extraction.
        """

        candidate_labels = event_types + ["Company"]
        result = self.zero_shot_classifier(sentence, candidate_labels)

        # Find the event type with the highest score
        event_type = max(
            result["labels"], key=lambda label: result["scores"][result["labels"].index(label)])
        if event_type not in event_types:
            event_type = EventType.O.value  # If no valid event type found

        company_name = self.extract_company_name(sentence)
        if not company_name:
            company_name = "Unknown"  # "Unknown" if no company name is found

        return {"event_type": event_type, "company": company_name}

    def get_model_response(self, prompt, max_length=512):
        logging.debug(f"Generating model response for prompt: {prompt}")
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Model response: {response}")
        return response

    def process_sentence(self, sentence):
        logging.info(f"Processing sentence: {sentence}")
        event_types = [e.value for e in EventType]

        extracted_events = self.extract_events_zero_shot(sentence, event_types)
        logging.info(f"Extracted events: {extracted_events}")
        return extracted_events

    def process_dataset(self, dataset_path):
        logging.info("Processing Data")
        logging.info(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, "r") as file:
            data = json.load(file)

        data = list(reversed(data))
        data = data[:100]

        results = []
        for item in data:
            sentence = item["sentence"][0]
            extracted_events = self.process_sentence(sentence)
            results.append(
                {"sentence": sentence, "extracted_events": extracted_events, "actual_events": item["events"]})

        logging.info("Processing Data Complete")
        logging.info("=" * 50)
        return results

    def save_results(self, results, output_path):
        logging.info(f"Saving results to: {output_path}")
        with open(output_path, "w") as file:
            json.dump(results, file, indent=4)

        logging.info("Results saved successfully")
        logging.info("=" * 50)

    def evaluate_events(self, results):
        """
        Evaluate the extracted events against the actual events in the dataset.
        """

        logging.info("Evaluating extracted events")

        y_true = []
        y_pred = []

        for result in results:
            actual_events = result["actual_events"]
            extracted_event = result["extracted_events"]

            if not actual_events:
                actual_events = [EventType.O.value]

            for actual, extract in zip(actual_events, extracted_event.values()):
                actual_event_enum = next(
                    (e for e in EventType if e.value == actual), EventType.O)
                y_true.append(actual_event_enum.value)

                extracted_event_enum = next(
                    (e for e in EventType if e.value == extracted_event["event_type"]), EventType.O)

                y_pred.append(extracted_event_enum.value)

        # Calculate Exact Match (EM)
        exact_matches = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        em_score = exact_matches / len(y_true)

        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[e.value for e in EventType], average=None)

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


def main():
    parser = argparse.ArgumentParser(description="Event Extraction Script")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-xl",
                        help="Model name to use for event extraction for ZeroShot Learning")
    parser.add_argument("--dataset_path", type=str,
                        default="EDT_dataset/Event_detection/train.json", help="Path to the dataset file")
    parser.add_argument("--output_file", type=str, default="extracted_events.json",
                        help="Name of the output file to save results")

    args = parser.parse_args()

    dataset_path = os.path.join(os.getcwd(), args.dataset_path)
    output_path = args.output_file
    model_name = args.model_name

    logging.info("Starting event extraction process")
    event_extractor = EventExtraction(model_name=model_name)
    results = event_extractor.process_dataset(dataset_path)
    event_extractor.save_results(results, output_path)

    logging.info("Extraction completed")
    evaluation_metrics = event_extractor.evaluate_events(results)
    logging.info(f"Evaluation Metrics: {evaluation_metrics}")


if __name__ == "__main__":
    main()
