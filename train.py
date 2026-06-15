import os
import pickle
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def clean_text(text):
    """
    Cleans the input text by converting it to lowercase and removing punctuation/special characters.
    Only alphanumeric characters and whitespace are kept.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep only letters, numbers, and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def main():
    print("--- PART 1: DATA PREPROCESSING ---")
    
    # 1. Load the dataset
    # The SMSSpamCollection file is a tab-separated values (TSV) file.
    dataset_path = 'SMSSpamCollection'
    print(f"Loading dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found! Please run the download script first.")
        
    df = pd.read_csv(dataset_path, sep='\t', names=['label', 'message'])
    print(f"Dataset loaded successfully with {len(df)} rows.")

    # 2. Check and handle missing values
    print("Checking for missing values in the dataset...")
    missing_counts = df.isnull().sum()
    print("Missing values per column:\n", missing_counts)
    
    # Drop rows with missing labels or messages if any exist
    if df.isnull().values.any():
        print("Handling missing values by dropping corrupted rows...")
        df = df.dropna().reset_index(drop=True)
        print(f"Cleaned dataset size: {len(df)} rows.")
    else:
        print("No missing values found.")

    # 3. Clean the text message column
    print("Cleaning text messages (lowercasing and removing special characters)...")
    df['clean_message'] = df['message'].apply(clean_text)
    
    # Remove any empty messages after cleaning
    df = df[df['clean_message'] != ""].reset_index(drop=True)
    
    # 4. Split dataset into training and testing sets (80% train, 20% test)
    # Using a fixed random_state for reproducibility
    print("Splitting dataset into 80% training and 20% testing sets...")
    X = df['clean_message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 5. Feature extraction using CountVectorizer
    # CountVectorizer converts text documents to a matrix of token counts.
    # Note: We fit the vectorizer ONLY on training data to prevent data leakage.
    print("Fitting CountVectorizer on the training set and transforming text to features...")
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} unique words.")

    print("\n--- PART 2: MODEL TRAINING & EVALUATION ---")
    
    # 6. Initialize and train Multinomial Naive Bayes model
    # Multinomial Naive Bayes is a classic and highly effective classifier for text word counts.
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    print("Model trained successfully.")

    # 7. Evaluate the model on the testing set
    print("Evaluating the model on testing data...")
    y_pred = model.predict(X_test_vec)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("Format explanation:")
    print(" [[True Ham,  False Spam]")
    print("  [False Ham, True Spam]]")
    
    print("\nClassification Report:")
    print(class_report)

    # 8. Save the model and vectorizer as pickle files
    # These pkl files will be loaded by the Flask application for prediction.
    model_filename = 'model.pkl'
    vectorizer_filename = 'vectorizer.pkl'
    
    print(f"Saving trained model to '{model_filename}'...")
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Saving CountVectorizer to '{vectorizer_filename}'...")
    with open(vectorizer_filename, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Training and exporting complete!")

if __name__ == '__main__':
    main()
