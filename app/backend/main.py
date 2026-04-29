# app/backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.core.model_registry import model_registry
from app.backend.routes.predict import router as predict_router
from app.backend.routes.compare import router as compare_router

app = FastAPI(title="EAGLE-Net API")

# Register routes
app.include_router(predict_router)
app.include_router(compare_router)

# CORS (safe for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models():
    """
    Load all models once when the server starts.
    """
    model_registry.load_all_models()


@app.get("/")
def root():
    return {"message": "EAGLE-Net backend running"}


@app.get("/health")
def health():
    return {"status": "ok"}