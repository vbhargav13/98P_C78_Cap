# ===== LESSON 13: Project Setup - AI Workspace =====
import pandas as pd              # pandas helps us work with tables of data (like a spreadsheet)
import string                   # string gives us tools to work with text and punctuation
import joblib                   # joblib lets us save and load our trained model easily

# ===== LESSON 14: Install Libraries + Text Processing Setup =====
import nltk                     # nltk is a library for working with human language
from nltk.stem import WordNetLemmatizer  # Helps reduce words to their base form (e.g., "running" → "run")

# ===== LESSON 21: Text to Numbers - Vectorizing Emotions =====
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into numbers

# ===== LESSON 22: Regression Concept (though used LogisticRegression here) =====
from sklearn.linear_model import LogisticRegression          # A basic ML model for classification

# ===== LESSON 23: Build Pipeline for Vectorization + Classification =====
from sklearn.pipeline import Pipeline       # Helps chain multiple ML steps

# ===== LESSON 20: Dataset Split for Training & Testing =====
from sklearn.model_selection import train_test_split  # Splits data into training and testing sets

# ===== LESSON 24: Evaluate Model Performance =====
from sklearn.metrics import classification_report  # Shows how well the model performed

# ===== LESSON 18: GUI Setup Preparation =====
import tkinter as tk          # tkinter is used for GUI windows and buttons
from tkinter import messagebox  # used to display alerts in GUI

# ===== LESSON 14: Download Required NLP Resources =====
nltk.download('punkt')     # Punkt helps break sentences into words
nltk.download('wordnet')   # WordNet helps in lemmatizing (simplifying) words

# ===== LESSON 15: Preprocess Text to Simplify Before Feeding to ML Model =====
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    """
    - Convert to lowercase
    - Remove punctuation
    - Tokenize into words
    - Lemmatize to root form
    - Rejoin for model use
    """
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    clean_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(clean_words)

# ===== LESSON 16: Load CSV Dataset & Clean It Using Preprocessing =====
def load_dataset():
    df = pd.read_csv("D:/ChromeDownload/enhanced_emotion_dataset.csv")
    df['text'] = df['text'].apply(preprocess_text)
    return df

# ===== LESSON 23: Model Training (Naive Bayes or Logistic Regression) =====
def train_model():
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),  # Convert text to numbers
        ('classifier', LogisticRegression(max_iter=300))     # Classify using Logistic Regression
    ])
    pipeline.fit(X_train, y_train)  # Train the pipeline
    y_pred = pipeline.predict(X_test)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump(pipeline, "emotion_pipeline.pkl")  # Save the model
    print("✅ Model pipeline saved as 'emotion_pipeline.pkl'.")

# ===== LESSON 24: Test Model with a GUI =====
def launch_gui():
    def predict():
        user_input = entry.get()
        if not user_input.strip():
            messagebox.showwarning("Input Required", "Please enter a sentence.")
            return
        clean_text = preprocess_text(user_input)
        try:
            model = joblib.load("emotion_pipeline.pkl")
            prediction = model.predict([clean_text])[0]
            probabilities = model.predict_proba([clean_text])[0]
            confidence = max(probabilities) * 100
            result_label.config(text=f"Predicted Emotion: {prediction} ({confidence:.2f}%)")
        except FileNotFoundError:
            result_label.config(text="⚠️ Model not found. Train the model first.")

    def clear_fields():
        entry.delete(0, tk.END)
        result_label.config(text="")

    window = tk.Tk()
    window.title("😊 AI Emotion Classifier 😊")
    window.geometry("450x300")
    window.config(bg="#E8F4FA")

    tk.Label(window, text="Enter a sentence:", font=("Arial", 12), bg="#E8F4FA").pack(pady=10)
    entry = tk.Entry(window, width=45, font=("Arial", 12))
    entry.pack(pady=5)

    button_frame = tk.Frame(window, bg="#E8F4FA")
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Predict Emotion", font=("Arial", 12), command=predict).grid(row=0, column=0, padx=10)
    tk.Button(button_frame, text="Clear", font=("Arial", 12), command=clear_fields).grid(row=0, column=1, padx=10)

    result_label = tk.Label(window, text="", font=("Arial", 14, "bold"), fg="#333333", bg="#E8F4FA")
    result_label.pack(pady=20)

    window.mainloop()

# ===== LESSON 17: Menu to Select Between Training and GUI =====
def main():
    while True:
        print("\n🎯 AI Emotion Classifier Menu")
        print("1. Train Model")
        print("2. Launch Emotion Predictor (UI)")
        print("3. Exit")
        choice = input("Choose an option: ")
        if choice == "1":
            train_model()
        elif choice == "2":
            launch_gui()
        elif choice == "3":
            print("👋 Exiting. Goodbye!")
            break
        else:
            print("❌ Invalid option. Try again.")

if __name__ == "__main__":
    main()
