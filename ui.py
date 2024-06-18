import joblib
import tkinter as tk
from tkinter import messagebox

# Load the saved Naive Bayes model and TF-IDF vectorizer
nb_model = joblib.load('naive_bayes_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_message():
    # Get user input
    user_message = entry.get()
    if user_message.strip() == "":
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    
    # Transform the message using the loaded TF-IDF vectorizer
    message_tfidf = tfidf_vectorizer.transform([user_message])
    
    # Predict the label using the loaded Naive Bayes model
    prediction = nb_model.predict(message_tfidf)
    pred_label = 'ham' if prediction[0] == 0 else 'spam'
    
    # Display the result
    messagebox.showinfo("Prediction Result", f"The message is predicted as: {pred_label}")

# Create the main window
root = tk.Tk()
root.title("SMS Spam Classifier")

# Create and place the input field
label = tk.Label(root, text="Enter your message:")
label.pack(pady=10)
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Create and place the predict button
predict_button = tk.Button(root, text="Predict", command=predict_message)
predict_button.pack(pady=20)

# Run the GUI event loop
root.mainloop()
