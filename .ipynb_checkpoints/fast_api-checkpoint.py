from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import torch

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained models and SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

# Define the models paths
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
# Load models
models = {
    "text_only": {name: joblib.load(path) for name, path in text_only_models.items()},
    "text_with_features": {name: joblib.load(path) for name, path in text_with_features_models.items()},
}

# Request model for prediction input
class PredictionRequest(BaseModel):
    product_title: str
    review_title: str
    review_text: str
    model_name: str
    include_features: bool = False
    helpful_vote: float = None
    rating: float = None

def preprocess_input(product_title, review_title, review_text, include_features, helpful_vote=None, rating=None):
    # Embed text data
    combined_text = f"product title: {product_title}. review title: {review_title}. review text: {review_text}"
    text_embedding = embedding_model.encode([combined_text], show_progress_bar=False)
    
    # Append numerical features if necessary
    if include_features:
        additional_features = np.array([[helpful_vote, rating]])
        full_features = np.concatenate([text_embedding, additional_features], axis=1)
    else:
        full_features = text_embedding
    return full_features

@app.post("/predict")
def predict_outlier(request: PredictionRequest):
    # Validate model selection
    if request.include_features:
        if request.model_name not in models["text_with_features"]:
            raise HTTPException(status_code=400, detail="Selected model does not support additional features.")
        model = models["text_with_features"][request.model_name]
    else:
        if request.model_name not in models["text_only"]:
            raise HTTPException(status_code=400, detail="Selected model does not support text-only input.")
        model = models["text_only"][request.model_name]

    # Validate additional features if required
    if request.include_features and (request.helpful_vote is None or request.rating is None):
        raise HTTPException(status_code=400, detail="Helpful vote and rating must be provided for models with additional features.")
    
    # Preprocess input
    features = preprocess_input(
        product_title=request.product_title,
        review_title=request.review_title,
        review_text=request.review_text,
        include_features=request.include_features,
        helpful_vote=request.helpful_vote,
        rating=request.rating
    )

    # Perform prediction
    prediction = model.predict(features)
    result = "Outlier" if prediction[0] == 1 else "Non-Outlier"

    return {"result": result, "model": request.model_name}

# Run with:
# uvicorn fast_api:app --reload
