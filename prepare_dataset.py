import os
import pandas as pd
import spacy
import torch
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Налаштування
INPUT_FILE = "properties.jsonl"
OUTPUT_DIR = "dataset_artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Завантаження української мовної моделі spaCy...")
nlp = spacy.load("uk_core_news_sm")


def preprocess_text(text):
    """Очищення та лематизація тексту"""
    doc = nlp(text.lower())
    clean_tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(clean_tokens)


def main():
    logging.info(f"Читання даних з {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)

    logging.info("Очищення текстів оголошень (це може зайняти хвилину)...")
    df["clean_desc"] = df["desc"].apply(preprocess_text)

    logging.info("Кодування категорій...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])

    # Зберігаємо LabelEncoder
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
    logging.info(f"Знайдено класи: {label_encoder.classes_}")

    logging.info("Створення TF-IDF векторів...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2,
    )
    X = vectorizer.fit_transform(df["clean_desc"]).toarray()

    # Зберігаємо Vectorizer (його словник), щоб обробляти нові тексти в майбутньому
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))

    logging.info("Розбиття на тренувальну та тестову вибірки...")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Конвертація у тензори PyTorch
    X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    logging.info("Збереження тензорів PyTorch...")
    torch.save(X_train_tensor, os.path.join(OUTPUT_DIR, "X_train.pt"))
    torch.save(y_train_tensor, os.path.join(OUTPUT_DIR, "y_train.pt"))
    torch.save(X_test_tensor, os.path.join(OUTPUT_DIR, "X_test.pt"))
    torch.save(y_test_tensor, os.path.join(OUTPUT_DIR, "y_test.pt"))

    logging.info(f"Всі артефакти збережено у папку '{OUTPUT_DIR}'.")
    logging.info(f"Розмір тренувальної вибірки: {X_train_tensor.shape[0]} оголошень.")
    logging.info(f"Розмір тестової вибірки: {X_test_tensor.shape[0]} оголошень.")


if __name__ == "__main__":
    main()
