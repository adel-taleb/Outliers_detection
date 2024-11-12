import tkinter as tk
from tkinter import messagebox
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

class OutliersTester:
    def __init__(self, text_only_models, text_with_features_models, embedding_model='all-MiniLM-L6-v2', device='cuda'):
        self.text_only_models = {name: joblib.load(path) for name, path in text_only_models.items()}
        self.text_with_features_models = {name: joblib.load(path) for name, path in text_with_features_models.items()}
        self.embedding_model = SentenceTransformer(embedding_model, device=device)

    def preprocess_input(self, product_title, review_title, review_text, include_features, helpful_vote=None, rating=None):
        combined_text = f"product title: {product_title}. review title: {review_title}. review text: {review_text}"
        text_embedding = self.embedding_model.encode([combined_text], show_progress_bar=False)
        
        if include_features:
            additional_features = np.array([[helpful_vote, rating]])
            full_features = np.concatenate([text_embedding, additional_features], axis=1)
        else:
            full_features = text_embedding
        return full_features

    def predict_outlier(self, model_name, product_title, review_title, review_text, include_features, helpful_vote=None, rating=None):
        full_features = self.preprocess_input(product_title, review_title, review_text, include_features, helpful_vote, rating)
        model = self.text_with_features_models[model_name] if include_features else self.text_only_models[model_name]
        prediction = model.predict(full_features)
        return bool(prediction[0] == 1)

# Define models paths
text_only_models =  {
    "Isolation Forest": "all_results/Health_and_Personal_Care/Isolation Forest/Isolation Forest.joblib",
    "AutoEncoder": "all_results/Health_and_Personal_Care/AutoEncoder/AutoEncoder.joblib",
    "Variational AutoEncoder (VAE)": "all_results/Health_and_Personal_Care/Variational AutoEncoder (VAE)/Variational AutoEncoder (VAE).joblib"
}
text_with_features_models = {
    "Isolation Forest": "all_results/Health_and_Personal_Care/Isolation Forest/Isolation Forest.joblib",
    "AutoEncoder": "all_results/Health_and_Personal_Care/AutoEncoder/AutoEncoder.joblib",
    "Variational AutoEncoder (VAE)": "all_results/Health_and_Personal_Care/Variational AutoEncoder (VAE)/Variational AutoEncoder (VAE).joblib"
}

# Initialize the tester
tester = OutliersTester(text_only_models=text_only_models, text_with_features_models=text_with_features_models)

# GUI setup
def toggle_features():
    if include_features_var.get():
        helpful_vote_entry.config(state=tk.NORMAL)
        rating_entry.config(state=tk.NORMAL)
        model_menu['menu'].delete(0, 'end')
        for model_name in text_with_features_models.keys():
            model_menu['menu'].add_command(label=model_name, command=tk._setit(model_var, model_name))
    else:
        helpful_vote_entry.config(state=tk.DISABLED)
        rating_entry.config(state=tk.DISABLED)
        model_menu['menu'].delete(0, 'end')
        for model_name in text_only_models.keys():
            model_menu['menu'].add_command(label=model_name, command=tk._setit(model_var, model_name))

def check_outlier():
    selected_model = model_var.get()
    product_title = product_title_entry.get()
    review_title = review_title_entry.get()
    review_text = review_text_entry.get("1.0", "end-1c")
    include_features = include_features_var.get()

    if include_features:
        try:
            helpful_vote = float(helpful_vote_entry.get())
            rating = float(rating_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Helpful Vote and Rating must be numbers.")
            return
    else:
        helpful_vote = None
        rating = None

    is_outlier = tester.predict_outlier(selected_model, product_title, review_title, review_text, include_features, helpful_vote, rating)
    result = "Outlier" if is_outlier else "Non-Outlier"
    messagebox.showinfo("Result", f"The review is classified as: {result} using {selected_model} model.")

# Create the main window
root = tk.Tk()
root.title("Outlier Detection Tester")
root.geometry("500x600")

# Model selection dropdown
tk.Label(root, text="Select Model:").pack()
model_var = tk.StringVar(root)
model_var.set("Select Model")
model_menu = tk.OptionMenu(root, model_var, *text_only_models.keys())
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

# Checkbox to include or exclude numerical features
include_features_var = tk.BooleanVar()
include_features_check = tk.Checkbutton(root, text="Include Helpful Vote and Rating", variable=include_features_var, command=toggle_features)
include_features_check.pack()

# Helpful Vote
tk.Label(root, text="Helpful Vote:").pack()
helpful_vote_entry = tk.Entry(root, width=20, state=tk.DISABLED)
helpful_vote_entry.pack()

# Rating
tk.Label(root, text="Rating:").pack()
rating_entry = tk.Entry(root, width=20, state=tk.DISABLED)
rating_entry.pack()

# Check button
check_button = tk.Button(root, text="Check Outlier", command=check_outlier)
check_button.pack(pady=10)

# Run the application
root.mainloop()
