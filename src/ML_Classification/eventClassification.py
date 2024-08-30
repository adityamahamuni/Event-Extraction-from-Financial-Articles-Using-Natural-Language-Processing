import re
import os
import argparse
import pandas as pd
import numpy as np
import string
import joblib
import logging
import json

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class EventClassificationModel:
    def __init__(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl', label_encoder_path='label_encoder.pkl'):
        """Initialize paths and variables for the model, vectorizer, and label encoder."""

        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        logging.info('Initialized EventClassificationModel')

    def load_data(self, file_path):
        """Load dataset from a JSON file."""

        data = pd.read_json(file_path)
        logging.info(f'Data loaded from {file_path}')
        return data

    def clean_text(self, text):
        """Clean and preprocess text by removing punctuation, stopwords, and applying lemmatization."""

        if isinstance(text, list):
            text = ' '.join(text)
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]

        cleaned_text = ' '.join(words)
        logging.debug(f'Cleaned text: {cleaned_text}')
        return cleaned_text

    def encode_labels(self, labels):
        """Encode labels into integers using LabelEncoder."""

        if isinstance(labels.iloc[0], list):
            labels = labels.apply(lambda x: x[0])
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        logging.info('Labels encoded')
        return encoded_labels

    def split_data(self, X, y):
        """Split data into training and testing sets."""

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        logging.info('Data split into train and test sets')
        return X_train, X_test, y_train, y_test

    def vectorize_text(self, train_texts, test_texts):
        """Convert text data into TF-IDF vectors."""

        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)
        logging.info('Text data vectorized')
        return X_train, X_test

    def perform_grid_search(self, X_train, y_train, model, param_grid):
        """Perform grid search to find the best hyperparameters for the model."""

        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        logging.info(f'Best parameters found: {best_params}')
        return best_params

    def perform_stratified_kfold(self, X, y, model, k=5):
        """Perform stratified k-fold cross-validation and return the mean accuracy."""

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        logging.info(f'Stratified k-fold scores: {scores}')
        logging.info(f'Mean accuracy: {np.mean(scores)}')
        return np.mean(scores)

    def train_models(self, X_train, y_train):
        """Train multiple models and select the best one based on cross-validation performance."""

        models = {
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier()
        }

        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            },
            'SVM': {
                'C': [0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [5, 10, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [5, 10, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }

        best_model = None
        best_score = 0
        best_params_dict = {}

        # Train each model and select the best one
        for model_name, model in models.items():
            logging.info(f'Training {model_name}')
            best_params = self.perform_grid_search(
                X_train, y_train, model, param_grids[model_name])
            model.set_params(**best_params)
            score = self.perform_stratified_kfold(X_train, y_train, model)
            best_params_dict[model_name] = best_params
            if score > best_score:
                best_score = score
                best_model = model

        best_model.fit(X_train, y_train)
        self.model = best_model
        logging.info(f'Best model selected: {best_model}')
        self.save_best_params(best_params_dict)
        return best_model

    def save_best_params(self, best_params_dict):
        """Save the best hyperparameters for each model to a JSON file."""

        best_params_path = os.path.join(
            os.path.dirname(self.model_path), 'best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(best_params_dict, f, indent=4)
        logging.info(f'Best hyperparameters saved to {best_params_path}')

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model on the test set and log the results."""

        y_pred = self.model.predict(X_test)
        y_test = self.label_encoder.inverse_transform(y_test)
        y_pred = self.label_encoder.inverse_transform(y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        logging.info(
            f'Evaluation Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
        logging.info(
            f'\nClassification Report:\n {classification_report(y_test, y_pred)}')

        return accuracy, precision, recall, f1

    def save_model(self):
        """Save the trained model, vectorizer, and label encoder to disk."""

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        joblib.dump(self.label_encoder, self.label_encoder_path)
        logging.info('Model, vectorizer, and label encoder saved')

    def load_model(self):
        """Load the trained model, vectorizer, and label encoder from disk."""

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.label_encoder = joblib.load(self.label_encoder_path)
        logging.info('Model, vectorizer, and label encoder loaded')

    def predict(self, new_texts):
        """Make predictions on new text data."""

        X_new = self.vectorizer.transform(new_texts)
        predictions = self.model.predict(X_new)
        logging.info(f'Predictions made for new texts: {new_texts}')
        return self.label_encoder.inverse_transform(predictions)

    def train_pipeline(self, file_path):
        """Complete pipeline for training, evaluating, and saving the model."""

        data = self.load_data(file_path)
        data['cleaned_sentence'] = data['sentence'].apply(self.clean_text)
        y = self.encode_labels(data['events'])

        X_train, X_test, y_train, y_test = self.split_data(
            data['cleaned_sentence'], y)
        X_train_vec, X_test_vec = self.vectorize_text(X_train, X_test)

        self.train_models(X_train_vec, y_train)
        self.evaluate_model(X_test_vec, y_test)

        self.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and use a text classification model.')
    parser.add_argument('--dataset_path', type=str,
                        help='Path to the dataset file')
    parser.add_argument('--model_name', type=str,
                        help='Name for saving the model')

    args = parser.parse_args()

    model_path = f"{args.model_name}.pkl"
    vectorizer_path = f"{args.model_name}_vectorizer.pkl"
    label_encoder_path = f"{args.model_name}_label_encoder.pkl"

    model = EventClassificationModel(
        model_path, vectorizer_path, label_encoder_path)
    model.train_pipeline(args.dataset_path)
