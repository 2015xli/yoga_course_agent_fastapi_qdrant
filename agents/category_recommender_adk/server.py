"""HTTP server for Category Recommender Agent compliant with Google ADK card definition."""
import sys, pathlib
# Ensure project root path available for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from recommend_course_from_category import CategoryCourseRecommender

app = FastAPI(title="Category Recommender Agent", version="0.1.0")

# ---- request / response schema -------------------------------------------------

class ComposeCourseRequest(BaseModel):
    user_query: str

class ComposeCourseResponse(BaseModel):
    sequence: List[str]

recommender: CategoryCourseRecommender | None = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = CategoryCourseRecommender(api_type="deepseek")
    try:
        yield
    finally:
        if recommender:
            recommender.close()

# Recreate FastAPI app with lifespan handler
app = FastAPI(title="Category Recommender Agent", version="0.1.0", lifespan=lifespan)

@app.post("/compose-course", response_model=ComposeCourseResponse)
async def compose_course(req: ComposeCourseRequest):
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialised")
    try:
        seq = recommender.recommend_course(req.user_query)
        return ComposeCourseResponse(sequence=seq)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    uvicorn.run("agents.category_recommender_adk.server:app", host="0.0.0.0", port=8002)
