import os
import json
import re
import logging
import argparse
from enum import Enum

import torch
from huggingface_hub import login
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

        self.model_name = model_name
        self.cache_dir = '/dcs/large/u5579267/.huggingface'

        if "flan-t5" in self.model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name, legacy=True, cache_dir=self.cache_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, cache_dir=self.cache_dir)
        elif "bart" in self.model_name.lower():
            self.tokenizer = BartTokenizer.from_pretrained(
                model_name, cache_dir=self.cache_dir)
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name, cache_dir=self.cache_dir)
        elif "gpt" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=self.cache_dir)
        elif "llama" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=self.cache_dir)
        else:
            print("Model not supported - running with pipeline")

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        logging.info("Initialized EventExtraction model")

    @staticmethod
    def initialize():
        token = os.getenv("HUGGINGFACE_TOKEN")
        login(token=token)
        logging.info("Initialized Hugging Face API")

    @staticmethod
    def extract_company_name(sentence):
        """Extracts the company name from the sentence based on common patterns."""

        pattern = r"\b[A-Z][a-zA-Z]*\s(?:Inc|Corp|Ltd|LLC|Group|Company|PLC|Corporation|Incorporated|N\.A\.)\b"
        match = re.search(pattern, sentence)
        if match:
            return match.group()
        return "Unknown"

    @staticmethod
    def get_schema_prompt(sentence):
        return f"""
            Extract event information from the following sentence and and return the most matching event as event_type

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

            Output:
        """

    @staticmethod
    def get_code_prompt(sentence):
        return f"""
            # Extract event information from the following sentences and return the most matching event as event_type
            Event = event_type

            events:= [Acquisition (A), Clinical Trial (CT), Regular Dividend (RD), Dividend Cut (DC), Dividend Increase (DI), Guidance Increase (GI), New Contract (NC), Reverse Stock Split (RSS), Special Dividend (SD), Stock Repurchase (SR), Stock Split (SS), Other/None (O).

            sentence:= "{sentence}"
            print(event_type)
        """

    @staticmethod
    def get_explanation_prompt(sentence):
        return f"""
            Extract event information from the following sentences and return the most matching event as event_type

            Event Type:
            - Acquisition (A): A company purchases another company or a significant portion of it.
            - Clinical Trial (CT): A company conducts a research study to test new medical treatments or drugs.
            - Regular Dividend (RD): A company distributes a portion of its earnings to shareholders regularly.
            - Dividend Cut (DC): A company reduces the amount of dividend it pays out to shareholders.
            - Dividend Increase (DI): A company increases the amount of dividend it pays out to shareholders.
            - Guidance Increase (GI): A company raises its future earnings or revenue forecast.
            - New Contract (NC): A company secures a new agreement for providing goods or services.
            - Reverse Stock Split (RSS): A company reduces the number of its outstanding shares to increase the share price.
            - Special Dividend (SD): A company makes a one-time distribution of additional earnings to shareholders.
            - Stock Repurchase (SR): A company buys back its own shares from the marketplace.
            - Stock Split (SS): A company increases the number of its outstanding shares by dividing its current shares.
            - Other/None (O): Events that do not fit into any of the specified categories.

            Sentence: "{sentence}"

            Output:

        """

    @staticmethod
    def get_pipeline_prompt(sentence):
        return f"""
            Stage 1:
            Extract event information from the following sentence and and return the most matching event as event_type

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

            Output:

            Stage 2:
            Explain the extracted event in detail and provide additional context
        """

    def set_prompt_type(self, prompt_type):
        self.prompt_type = prompt_type

    def get_prompt(self, sentence):
        """
        Get the appropriate prompt based on the prompt type.
        """

        if self.prompt_type == "schema":
            return self.get_schema_prompt(sentence)
        elif self.prompt_type == "code":
            return self.get_code_prompt(sentence)
        elif self.prompt_type == "explanation":
            return self.get_explanation_prompt(sentence)
        elif self.prompt_type == "pipeline":
            return self.get_pipeline_prompt(sentence)
        else:
            raise ValueError(f"Invalid prompt type: {self.prompt_type}")

    def get_model_response(self, prompt, max_new_tokens=512):
        """
        Generate a response from the model given a prompt.
        """

        logging.debug(f"Generating model response for prompt: {prompt}")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=max_new_tokens, truncation=True).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():  # Mixed precision
            outputs = self.model.generate(
                **inputs, max_length=inputs['input_ids'].shape[1] + max_new_tokens, max_new_tokens=200)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.debug(f"Model response: {response}")
        return response

    def extract_event_type(self, text):
        pattern = re.compile(r'(?<=event_type": ")(.*?)(?=")', re.IGNORECASE)
        match = pattern.search(text)

        if match:
            event_type = match.group(1)
            return event_type
        else:
            return "N/A"

    def process_sentence(self, sentence):
        print(f"Processing sentence: {sentence}")
        prompt = self.get_prompt(sentence)
        response = self.get_model_response(prompt, max_new_tokens=1024)

        if "llama" in self.model_name.lower():
            event_type = self.extract_event_type(response)
        else:
            event_type = response

        return event_type

    def process_dataset(self, dataset_path):
        """
        Process the dataset and extract events from the sentences
        """
        logging.info(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, "r") as file:
            data = json.load(file)

        results = []
        for item in data:
            sentence = item["sentence"][0]
            extracted_events = self.process_sentence(sentence)
            results.append(
                {"sentence": sentence, "extracted_events": extracted_events, "actual_events": item["events"]})

            torch.cuda.empty_cache()

            logging.info(f"Extracted events: {extracted_events}")

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
        Evaluate the extracted events using Exact Match (EM) and F1 Score
        """
        logging.info("Evaluating extracted events")

        y_true = []
        y_pred = []

        for result in results:
            actual_events = result["actual_events"]
            extracted_event = result["extracted_events"]

            if not actual_events:
                actual_events = [EventType.O.value]

            for actual, extract in zip(actual_events, extracted_event):
                actual_event_enum = next(
                    (e for e in EventType if e.value == actual), EventType.O)
                y_true.append(actual_event_enum.value)

                extracted_event_enum = next(
                    (e for e in EventType if e.value == extracted_event), None)

                y_pred.append(extracted_event_enum.value)

        # Calculate Exact Match (EM)
        exact_matches = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        em_score = exact_matches / len(y_true)

        # Calculate F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted')

        logging.info(
            f"Evaluation metrics - Exact Match (EM): {em_score}, F1 Score: {f1}")
        logging.info("=" * 50)
        return {"exact_match": em_score, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Event Extraction Script")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-xl",
                        help="Model name to use for event extraction")
    parser.add_argument("--dataset_path", type=str,
                        default="EDT_dataset/Event_detection/train.json", help="Path to the dataset file")
    parser.add_argument("--output_file", type=str, default="extracted_events.json",
                        help="Name of the output file to save results")

    args = parser.parse_args()

    dataset_path = os.path.join(os.getcwd(), args.dataset_path)
    model_name = args.model_name

    logging.info("Starting event extraction process")
    event_extractor = EventExtraction(model_name=model_name)
    event_extractor.initialize()

    prompt_types = ["schema", "code", "explanation", "pipeline"]

    for prompt_type in prompt_types:
        logging.info(f"Prompt Type: {prompt_type}")
        event_extractor.set_prompt_type(prompt_type)

        output_file = args.output_file
        output_file = output_file.replace(".json", f"_{prompt_type}.json")
        output_path = os.path.join(
            os.getcwd(), "EDT_dataset", "Event_detection", output_file)

        results = event_extractor.process_dataset(dataset_path)
        event_extractor.save_results(results, output_path)

        logging.info(
            "Extraction completed for prompt type : {prompt_type} - Results saved to {output_path} - Evaluating results")
        evaluation_metrics = event_extractor.evaluate_events(results)

        logging.info(f"Evaluation Metrics {prompt_type}: {evaluation_metrics}")
        logging.info("=" * 50)

    logging.info("Zeroshot Event Extraction Process Completed")


if __name__ == "__main__":
    main()
