from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import torch

# Initialize FastAPI app and Jinja2 templates
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, (pour fonctionnement en wsl)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
sub_dataset = "Health_and_Personal_Care"
# Load pre-trained models and embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
text_only_models =  {
    "Isolation Forest": "all_results/"+sub_dataset+"/Isolation Forest/Isolation Forest.joblib",
    "AutoEncoder": "all_results/"+sub_dataset+"/AutoEncoder/AutoEncoder.joblib",
    "Variational AutoEncoder (VAE)": "all_results/"+sub_dataset+"/Variational AutoEncoder (VAE)/Variational AutoEncoder (VAE).joblib"
}
text_with_features_models = {
    "Isolation Forest with rates and helpful score": "all_results/"+sub_dataset+"/Isolation Forest_with_numerical_data/Isolation Forest.joblib",
    "AutoEncoder with rates and helpful score": "all_results/"+sub_dataset+"/AutoEncoder_with_numerical_data/AutoEncoder.joblib",
    "Variational AutoEncoder (VAE) with rates and helpful score": "all_results/"+sub_dataset+"/Variational AutoEncoder (VAE)_with_numerical_data/Variational AutoEncoder (VAE).joblib"
}
models = {
    "text_only": {name: joblib.load(path) for name, path in text_only_models.items()},
    "text_with_features": {name: joblib.load(path) for name, path in text_with_features_models.items()},
}
scaler = joblib.load("all_results/"+sub_dataset+"/numerical_scaler.joblib")

class PredictionRequest(BaseModel):
    product_title: str
    review_title: str
    review_text: str
    model_name: str
    include_features: bool = False
    helpful_vote: float = None
    rating: float = None

def preprocess_input(product_title, review_title, review_text, include_features, helpful_vote=None, rating=None):
    combined_text = f"product title: {product_title}. review title: {review_title}. review text: {review_text}"
    text_embedding = embedding_model.encode([combined_text], show_progress_bar=False)
    if include_features:
        # Scale the helpful_vote and rating using the preloaded scaler
        additional_features = np.array([[helpful_vote, rating]])
        scaled_features = scaler.transform(additional_features)
        # Concatenate scaled features with the text embedding
        full_features = np.concatenate([text_embedding, scaled_features], axis=1)
    else:
        full_features = text_embedding
    return full_features

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "text_only_models": list(text_only_models.keys()), "text_with_features_models": list(text_with_features_models.keys())})

@app.post("/predict")
async def predict_outlier(request: PredictionRequest):
    if request.include_features:
        if request.model_name not in models["text_with_features"]:
            raise HTTPException(status_code=400, detail="Selected model does not support additional features.")
        model = models["text_with_features"][request.model_name]
    else:
        if request.model_name not in models["text_only"]:
            raise HTTPException(status_code=400, detail="Selected model does not support text-only input.")
        model = models["text_only"][request.model_name]

    if request.include_features and (request.helpful_vote is None or request.rating is None):
        raise HTTPException(status_code=400, detail="Helpful vote and rating must be provided for models with additional features.")
    
    features = preprocess_input(
        product_title=request.product_title,
        review_title=request.review_title,
        review_text=request.review_text,
        include_features=request.include_features,
        helpful_vote=request.helpful_vote,
        rating=request.rating
    )

    prediction = model.predict(features)
    result = "Outlier" if prediction[0] == 1 else "Non-Outlier"

    return {"result": result, "model": request.model_name}
