import tkinter as tk
from tkinter import messagebox
from sentence_transformers import SentenceTransformer
import joblib
import os
from outliers_tester import OutliersTester
class OutliersTester:
    def __init__(self, model_paths, embedding_model='all-MiniLM-L6-v2', device='cuda'):
        # Dictionary of models
        self.models = {name: joblib.load(path) for name, path in model_paths.items()}
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(embedding_model, device=device)

    def preprocess_input(self, product_title, review_title, review_text):
        # Combine the input text
        combined_text = f"product title: {product_title}. review title: {review_title}. review text: {review_text}"
        # Generate embeddings
        embeddings = self.embedding_model.encode([combined_text], show_progress_bar=False)
        return embeddings

    def predict_outlier(self, model_name, product_title, review_title, review_text):
        # Preprocess the input
        embeddings = self.preprocess_input(product_title, review_title, review_text)
        # Select model and predict outlier label (1 = outlier, 0 = inlier)
        model = self.models[model_name]
        prediction = model.predict(embeddings)
        return bool(prediction[0] == 1)

# Dictionary of model paths (add paths to each model you want to use)
model_paths = {
    "Isolation Forest": "all_results/Health_and_Personal_Care/Isolation Forest/Isolation Forest.joblib",
    "AutoEncoder": "all_results/Health_and_Personal_Care/AutoEncoder/AutoEncoder.joblib",
    "Variational AutoEncoder (VAE)": "all_results/Health_and_Personal_Care/Variational AutoEncoder (VAE)/Variational AutoEncoder (VAE).joblib"
}

# Initialize tester
tester = OutliersTester(model_paths=model_paths)

# GUI setup
def check_outlier():
    # Get input from the text fields
    selected_model = model_var.get()
    product_title = product_title_entry.get()
    review_title = review_title_entry.get()
    review_text = review_text_entry.get("1.0", "end-1c")

    # Predict if it's an outlier
    is_outlier = tester.predict_outlier(selected_model, product_title, review_title, review_text)
    
    # Display the result
    result = "Outlier" if is_outlier else "Non-Outlier"
    messagebox.showinfo("Result", f"The review is classified as: {result} using {selected_model} model.")

# Create the main window
root = tk.Tk()
root.title("Outlier Detection Tester")
root.geometry("500x450")

# Model selection dropdown
tk.Label(root, text="Select Model:").pack()
model_var = tk.StringVar(root)
model_var.set("Isolation Forest")  # Set default model
model_menu = tk.OptionMenu(root, model_var, *model_paths.keys())
model_menu.pack()

# Product title
tk.Label(root, text="Product Title:").pack()
product_title_entry = tk.Entry(root, width=50)
product_title_entry.pack()

# Review title
tk.Label(root, text="Review Title:").pack()
review_title_entry = tk.Entry(root, width=50)
review_title_entry.pack()

# Review text
tk.Label(root, text="Review Text:").pack()
review_text_entry = tk.Text(root, height=10, width=50)
review_text_entry.pack()

# Check button
check_button = tk.Button(root, text="Check Outlier", command=check_outlier)
check_button.pack(pady=10)

# Run the application
root.mainloop()
