import argparse
import dataclasses
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import uvicorn

from check_yoga_pose import YogaPoseChecker

# --- Data Structures ---
# These define the API contract for our service.

@dataclasses.dataclass
class CheckPoseRequest:
    pose_name: str
    user_query: str

@dataclasses.dataclass
class CheckPoseResponse:
    final_pose_name: str | None
    was_replaced: bool

# This will hold the global instance of our checker logic.
# It will be initialized within the lifespan context.
yoga_pose_checker_instance: YogaPoseChecker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    """
    global yoga_pose_checker_instance
    # Initialize the YogaPoseChecker instance
    # The api_type will be passed from the main function via app.state
    api_type = app.state.api_type
    yoga_pose_checker_instance = YogaPoseChecker(api_type=api_type)
    logging.info("YogaPoseChecker initialized.")
    yield
    # Clean up resources on shutdown
    if yoga_pose_checker_instance:
        yoga_pose_checker_instance.close()
    logging.info("YogaPoseChecker resources have been shut down.")

# --- FastAPI Application ---

app = FastAPI(
    title="Yoga Pose Checker Service",
    description="An API service to check if a yoga pose is suitable for a user and find a replacement if not.",
    lifespan=lifespan
)

@app.post("/check-pose", response_model=CheckPoseResponse)
async def check_pose_endpoint(request: CheckPoseRequest) -> CheckPoseResponse:
    """
    The main API endpoint. It receives a pose and query, checks for contraindications,
    and returns a validated pose or a replacement.
    """
    if not yoga_pose_checker_instance:
        raise HTTPException(status_code=503, detail="Service not available")

    logging.info(f"Received request to check pose: {request.pose_name}")
    
    original_pose = request.pose_name
    final_pose = yoga_pose_checker_instance.check_and_replace_pose(
        pose_name=original_pose,
        user_query=request.user_query
    )
    
    was_replaced = original_pose != final_pose and final_pose is not None
    
    logging.info(f"Check complete. Final pose: {final_pose}, Replaced: {was_replaced}")
    
    return CheckPoseResponse(
        final_pose_name=final_pose,
        was_replaced=was_replaced
    )

def main():
    """
    Main entry point to start the server.
    """
    parser = argparse.ArgumentParser(description="Yoga Pose Checker API Server")
    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "deepseek"],
        default="deepseek",
        help="Specify which model API to use (openai or deepseek).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080, # A default port
        help="The port for the API server to listen on.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="The host for the API server to bind to.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Store api_type in app.state for access in lifespan
    app.state.api_type = args.api

    logging.info(f"🧘 Starting Yoga Pose Checker API Server on http://{args.host}:{args.port}")
    
    # Start the Uvicorn server programmatically.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
