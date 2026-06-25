"""
GenAI Assessment Engine — FastAPI Backend
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.assessment import router as assessment_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

app = FastAPI(
    title="GenAI Assessment Engine",
    description="Adaptive assessment generation using HuggingFace models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "GenAI Assessment Engine"}


@app.get("/")
async def root():
    return {"message": "GenAI Assessment Engine API. See /docs for endpoints."}
