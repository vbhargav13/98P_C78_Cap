import pandas as pd              # pandas helps us work with tables of data (like a spreadsheet)
import string                   # string gives us tools to work with text and punctuation
import joblib                   # joblib lets us save and load our trained model easily
import nltk                     # nltk is a library for working with human language
from nltk.stem import WordNetLemmatizer  # This helps reduce words to their base form (e.g., "running" → "run")
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into numbers our model can understand
from sklearn.linear_model import LogisticRegression          # A simple but powerful machine learning model
from sklearn.pipeline import Pipeline       # Helps us string together multiple steps (vectorize + classify)
from sklearn.model_selection import train_test_split  # Splits our data into a part for training and a part for testing
from sklearn.metrics import classification_report  # Shows how well our model is doing
import tkinter as tk          # tkinter is the library for building graphical user interfaces (windows, buttons, etc.)
from tkinter import messagebox  # messagebox shows pop-up messages in our GUI

# ====== NLTK Downloads (only needed the first time you run this) ======
nltk.download('punkt')     # Download the Punkt tokenizer (breaks sentences into words)
nltk.download('wordnet')   # Download the WordNet database (used for lemmatization)

# ====== PREPROCESSING SETUP ======
lemmatizer = WordNetLemmatizer()  # Create a lemmatizer to turn words into their base form

def preprocess_text(text):
    """
    1. Convert the text to lowercase so 'Happy' and 'happy' are treated the same.
    2. Remove all punctuation (like commas, periods) so words are clean.
    3. Split the text into individual words.
    4. Lemmatize each word (e.g., 'running' → 'run') to simplify the text.
    5. Join the words back into a single string.
    """
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    clean_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(clean_words)

# ====== LOAD DATASET ======
def load_dataset():
    """
    1. Read the CSV file that contains our example sentences and their emotions.
    2. Apply our preprocessing to clean every sentence.
    3. Return the cleaned-up DataFrame.
    """
    df = pd.read_csv("D:/ChromeDownload/enhanced_emotion_dataset.csv")
    df['text'] = df['text'].apply(preprocess_text)
    return df

# ====== TRAIN THE MODEL ======
def train_model():
    """
    1. Load and preprocess the data.
    2. Split into training and testing sets (80% train, 20% test).
    3. Create a pipeline:
       - TfidfVectorizer: turns text into numeric features.
       - LogisticRegression: classifies the emotion.
    4. Fit (train) the pipeline on the training data.
    5. Test the model on unseen data and print a report.
    6. Save the trained pipeline to a file for later use.
    """
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['emotion'],
        test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=300))
    ])
    pipeline.fit(X_train, y_train)  # Train the model

    # Predict on the test set and show performance
    y_pred = pipeline.predict(X_test)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save our trained pipeline so we don't need to train again
    joblib.dump(pipeline, "emotion_pipeline.pkl")
    print("✅ Model pipeline saved as 'emotion_pipeline.pkl'.")

# ====== LAUNCH THE GRAPHICAL USER INTERFACE (GUI) ======
def launch_gui():
    """
    1. Create a window with tkinter.
    2. Add an entry box for the user to type a sentence.
    3. Add buttons: Predict Emotion and Clear.
    4. When Predict is clicked:
       - Preprocess the input.
       - Load the saved model.
       - Predict emotion and confidence.
       - Show the result in the window.
    5. When Clear is clicked:
       - Erase the entry box and result label.
    """
    def predict():
        user_input = entry.get()
        if not user_input.strip():  # Check if the input is empty
            messagebox.showwarning("Input Required", "Please enter a sentence.")
            return

        clean_text = preprocess_text(user_input)
        try:
            model = joblib.load("emotion_pipeline.pkl")
            prediction = model.predict([clean_text])[0]
            probabilities = model.predict_proba([clean_text])[0]
            confidence = max(probabilities) * 100
            result_label.config(
                text=f"Predicted Emotion: {prediction} ({confidence:.2f}%)"
            )
        except FileNotFoundError:
            result_label.config(
                text="⚠️ Model not found. Train the model first."
            )

    def clear_fields():
        entry.delete(0, tk.END)      # Clear the text entry
        result_label.config(text="")  # Clear the result label

    # Set up the main window
    window = tk.Tk()
    window.title("😊 AI Emotion Classifier 😊")
    window.geometry("450x300")
    window.config(bg="#E8F4FA")  # Light blue background for a friendly look

    # Instruction label
    tk.Label(
        window,
        text="Enter a sentence:",
        font=("Arial", 12),
        bg="#E8F4FA"
    ).pack(pady=10)

    # Text entry box
    entry = tk.Entry(window, width=45, font=("Arial", 12))
    entry.pack(pady=5)

    # Frame to hold buttons side by side
    button_frame = tk.Frame(window, bg="#E8F4FA")
    button_frame.pack(pady=10)

    tk.Button(
        button_frame,
        text="Predict Emotion",
        font=("Arial", 12),
        command=predict
    ).grid(row=0, column=0, padx=10)

    tk.Button(
        button_frame,
        text="Clear",
        font=("Arial", 12),
        command=clear_fields
    ).grid(row=0, column=1, padx=10)

    # Label to display the prediction result
    result_label = tk.Label(
        window,
        text="",
        font=("Arial", 14, "bold"),
        fg="#333333",
        bg="#E8F4FA"
    )
    result_label.pack(pady=20)

    window.mainloop()  # Start the GUI event loop

# ====== MAIN MENU IN TERMINAL ======
def main():
    """
    Provide a simple text menu in the terminal:
    1. Train the model
    2. Launch the GUI to predict emotions
    3. Exit the program
    """
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
