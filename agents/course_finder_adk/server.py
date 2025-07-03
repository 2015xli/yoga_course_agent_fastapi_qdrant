"""HTTP server for Course Finder Agent compliant with Google ADK card definition."""
import sys, pathlib
# Ensure project root is in PYTHONPATH for module imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from get_course_candidates_for_query import CourseFinder

app = FastAPI(title="Course Finder Agent", version="0.1.0")

# ---- request / response schemas -------------------------------------------------

class FindCoursesRequest(BaseModel):
    user_query: str

class FindCoursesResponse(BaseModel):
    courses: List[str]

# -------------------------------------------------------------------------------

finder: CourseFinder | None = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global finder
    finder = CourseFinder(api_type="deepseek")
    try:
        yield
    finally:
        if finder:
            finder.close()

# Recreate the FastAPI app with lifespan handler
app = FastAPI(title="Course Finder Agent", version="0.1.0", lifespan=lifespan)

@app.post("/find-courses", response_model=FindCoursesResponse)
async def find_courses(req: FindCoursesRequest):
    if not finder:
        raise HTTPException(status_code=503, detail="Finder not initialised")
    try:
        course_names = finder.find_candidates(req.user_query)
        return FindCoursesResponse(courses=course_names)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    uvicorn.run("agents.course_finder_adk.server:app", host="0.0.0.0", port=8001)
