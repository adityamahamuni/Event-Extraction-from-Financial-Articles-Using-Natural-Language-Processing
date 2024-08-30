import re
import os
import argparse
import pandas as pd
import numpy as np
import string
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class EventDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, embedding_matrix):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        dense_outputs = self.fc(hidden[-1])
        return dense_outputs


class EventClassificationModel:
    def __init__(self, model_path='model.pth', tokenizer_path='tokenizer.pkl', label_encoder_path='label_encoder.pkl'):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path

        self.glove_dir = os.path.join(os.getcwd(), 'glove.6B')

        # Model, tokenizer, and label encoder placeholders
        self.model = None
        self.tokenizer = None
        self.label_encoder = None

        # Configuration parameters
        self.max_sequence_length = 100
        self.embedding_dim = 100
        self.hidden_dim = 128
        self.n_layers = 2
        self.output_dim = None

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

        # Set the output dimension based on the number of classes
        self.output_dim = len(self.label_encoder.classes_)
        return encoded_labels

    def split_data(self, X, y):
        """Split data into training and testing sets."""

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        logging.info('Data split into train and test sets')
        return X_train, X_test, y_train, y_test

    def create_tokenizer(self, texts):
        """Create a tokenizer to map words to integers."""

        self.tokenizer = {}
        index = 1
        for text in texts:
            for word in text.split():
                if word not in self.tokenizer:
                    self.tokenizer[word] = index
                    index += 1
        logging.info('Tokenizer created and fitted on texts')

    def texts_to_sequences(self, texts):
        """Convert text to sequences of integers using the tokenizer."""

        sequences = []
        for text in texts:
            seq = [self.tokenizer.get(word, 0) for word in text.split()]
            sequences.append(seq)
        padded_sequences = self.pad_sequences(sequences)
        return padded_sequences

    def pad_sequences(self, sequences):
        """Pad sequences to ensure uniform length."""

        padded_sequences = np.zeros(
            (len(sequences), self.max_sequence_length), dtype=int)
        for i, seq in enumerate(sequences):
            if len(seq) > self.max_sequence_length:
                padded_sequences[i] = np.array(seq[:self.max_sequence_length])
            else:
                padded_sequences[i, -len(seq):] = np.array(seq)
        return padded_sequences

    def load_glove_embeddings(self):
        """Load pre-trained GloVe embeddings."""

        embeddings_index = {}
        with open(os.path.join(self.glove_dir, 'glove.6B.100d.txt'), encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        logging.info('GloVe embeddings loaded')
        return embeddings_index

    def create_embedding_matrix(self, word_index, embeddings_index):
        """Create an embedding matrix from GloVe embeddings for the vocabulary."""

        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
        return embedding_matrix

    def build_lstm_model(self, vocab_size, embedding_matrix):
        """Build the LSTM model for text classification."""

        model = LSTMClassifier(vocab_size, self.embedding_dim, self.hidden_dim,
                               self.output_dim, self.n_layers, embedding_matrix)
        logging.info('LSTM model built')
        return model

    def train_lstm_model(self, X_train, y_train):
        """Train the LSTM model on the training data."""

        embeddings_index = self.load_glove_embeddings()
        word_index = self.tokenizer
        embedding_matrix = self.create_embedding_matrix(
            word_index, embeddings_index)
        vocab_size = len(word_index) + 1

        self.model = self.build_lstm_model(vocab_size, embedding_matrix)

        # Prepare DataLoader for training
        train_dataset = EventDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Ensure y_train contains integers
        y_train = y_train.to(torch.int)

        # Calculate class weights for handling class imbalance
        class_sample_count = np.array(
            [len(np.where(y_train.numpy() == t)[0]) for t in np.unique(y_train.numpy())])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train.numpy()])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        # Re-define DataLoader with WeightedRandomSampler
        train_loader = DataLoader(
            train_dataset, batch_size=64, sampler=sampler)

        # Using CrossEntropyLoss for multiclass classification
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        self.model.train()
        for epoch in range(5):
            total_loss = 0
            for sequences, labels in train_loader:
                optimizer.zero_grad()
                output = self.model(sequences)
                loss = criterion(output, labels.long())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(
                f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')
        logging.info('LSTM model trained')

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model on the test data."""

        test_dataset = EventDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for sequences, _ in test_loader:
                output = self.model(sequences)
                preds = torch.argmax(output, dim=1).cpu().numpy()
                y_pred.extend(preds)

        y_pred = np.array(y_pred).flatten()
        y_test = y_test.cpu().numpy()

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
        """Save the trained model, tokenizer, and label encoder."""

        torch.save(self.model.state_dict(), self.model_path)
        with open(self.tokenizer_path, 'wb') as handle:
            joblib.dump(self.tokenizer, handle)
        joblib.dump(self.label_encoder, self.label_encoder_path)
        logging.info('Model, tokenizer, and label encoder saved')

    def load_model(self):
        """Load the model, tokenizer, and label encoder from disk."""

        embeddings_index = self.load_glove_embeddings()
        word_index = self.tokenizer
        embedding_matrix = self.create_embedding_matrix(
            word_index, embeddings_index)
        vocab_size = len(word_index) + 1

        self.model = self.build_lstm_model(vocab_size, embedding_matrix)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        with open(self.tokenizer_path, 'rb') as handle:
            self.tokenizer = joblib.load(handle)
        self.label_encoder = joblib.load(self.label_encoder_path)
        logging.info('Model, tokenizer, and label encoder loaded')

    def predict(self, new_texts):
        """Make predictions on new text data."""

        X_new = self.texts_to_sequences(new_texts)
        X_new = torch.tensor(X_new, dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X_new)
            predictions = torch.argmax(output, dim=1).cpu().numpy()

        predictions = self.label_encoder.inverse_transform(predictions)
        logging.info(f'Predictions made for new texts: {new_texts}')
        return predictions

    def train_pipeline(self, file_path):
        """Complete pipeline for training, evaluating, and saving the model."""

        data = self.load_data(file_path)
        data['cleaned_sentence'] = data['sentence'].apply(self.clean_text)
        y = self.encode_labels(data['events'])

        X_train, X_test, y_train, y_test = self.split_data(
            data['cleaned_sentence'], y)
        self.create_tokenizer(X_train)

        X_train_seq = self.texts_to_sequences(X_train)
        X_test_seq = self.texts_to_sequences(X_test)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        self.train_lstm_model(X_train_seq, y_train)

        accuracy, precision, recall, f1 = self.evaluate_model(
            X_test_seq, y_test)

        self.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and use a text classification model.')
    parser.add_argument('--dataset_path', type=str,
                        help='Path to the dataset file')
    parser.add_argument('--model_name', type=str,
                        help='Name for saving the model')

    args = parser.parse_args()

    model_path = f"{args.model_name}.pth"
    tokenizer_path = f"{args.model_name}_tokenizer.pkl"
    label_encoder_path = f"{args.model_name}_label_encoder.pkl"

    model = EventClassificationModel(
        model_path, tokenizer_path, label_encoder_path)
    model.train_pipeline(args.dataset_path)
