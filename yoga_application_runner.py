import argparse
import logging
import subprocess
import time
import requests
import re
import glob
import yaml
from pathlib import Path

# For Neo4j detail retrieval reuse logic from existing agent
from yoga_models import PoseInSequence, CourseCandidate

# --- ADK Card discovery helpers -------------------------------------------------

def load_card_info(card_path: Path) -> dict:
    """Loads a YAML card and returns parsed dict."""
    with open(card_path, "r") as f:
        return yaml.safe_load(f)


def discover_agent_cards() -> dict[str, dict]:
    """Return mapping of action_id to tuple(card_info, agent_dir Path)."""
    mapping = {}
    for card_file in glob.glob("agents/*_adk/card.yaml"):
        dir_path = Path(card_file).parent
        card = load_card_info(Path(card_file))
        for action in card.get("actions", []):
            mapping[action["id"]] = {
                "card": card,
                "agent_dir": dir_path,
                "endpoint": action["http"]["endpoint"].rstrip("/"),
            }
    return mapping


def start_agent_server(agent_dir: Path):
    """Launch the server.py inside *agent_dir*; return the detected base URL."""
    cmd = ["python", str(agent_dir / "server.py")]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base_url = None
    for line in iter(proc.stdout.readline, ''):
        logging.info(f"[{agent_dir.name}]: {line.strip()}")
        m = re.search(r"Uvicorn running on (http://[0-9\.:]+)", line)
        if m:
            base_url = m.group(1)
            break
        if proc.poll() is not None:
            logging.error(f"{agent_dir.name} server terminated unexpectedly.")
            break
    if base_url is None:
        raise RuntimeError(f"Could not determine address for {agent_dir.name} server.")
    return proc, base_url


class YogaApplicationRunner:
    """
    Orchestrates the yoga recommendation process by coordinating with external ADK agents and the pose checker service.
    """
    def __init__(self, pose_api_base: str, course_finder_url: str, category_url: str):
        self.pose_api_base = pose_api_base.rstrip("/")
        self.course_finder_url = course_finder_url.rstrip("/")
        self.category_url = category_url.rstrip("/")

        # Neo4j driver for detail retrieval
        import os
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
        from neo4j import GraphDatabase
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def _validate_sequence(self, sequence: list[str], user_query: str) -> list[str] | None:
        """
        Validates a sequence of poses by calling the pose checker API.

        Returns:
            A validated list of pose names, or None if the sequence is unacceptable.
        """
        validated_sequence = []
        removed_poses_count = 0
        max_removals_allowed = 2
        check_url = f"{self.pose_api_base}/check-pose"

        for pose_name in sequence:
            try:
                payload = {"pose_name": pose_name, "user_query": user_query}
                logging.info(f"Attempting to check pose '{pose_name}' via API: {check_url} with payload {payload}")
                response = requests.post(check_url, json=payload, timeout=45)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                result = response.json()
                final_pose_name = result.get("final_pose_name")

                if final_pose_name:
                    validated_sequence.append(final_pose_name)
                    if result.get("was_replaced"):
                        logging.info(f"Pose '{pose_name}' was replaced with '{final_pose_name}'.")
                else:
                    removed_poses_count += 1
                    logging.warning(f"Pose '{pose_name}' was unsuitable and removed (no replacement found).")

            except requests.exceptions.ConnectionError as e:
                logging.error(f"Connection error checking pose '{pose_name}'. Is the server running at {check_url}? Error: {e}")
                removed_poses_count += 1
            except requests.exceptions.Timeout as e:
                logging.error(f"Timeout checking pose '{pose_name}'. Server took too long to respond. Error: {e}")
                removed_poses_count += 1
            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP error checking pose '{pose_name}': {e}. Response: {e.response.text}")
                removed_poses_count += 1
            except requests.exceptions.RequestException as e:
                logging.error(f"General request error checking pose '{pose_name}': {e}. It will be removed.")
                removed_poses_count += 1
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing pose '{pose_name}': {e}. It will be removed.")
                removed_poses_count += 1

        if removed_poses_count > max_removals_allowed:
            logging.error(f"Course rejected: {removed_poses_count} poses were removed, which is more than the allowed {max_removals_allowed}.")
            return None
        
        return validated_sequence

    def _get_course_details(self, course_names: list[str]) -> list[CourseCandidate]:
        """Fetch full course info (description, sequence) from Neo4j."""
        query = """
        UNWIND $course_names AS course_name
        MATCH (c:Course {id: course_name})-[rel:INCLUDES_POSE]->(p:Pose)
        WITH c, rel, p
        ORDER BY rel.order
        RETURN c.id AS name,
               c.description AS description,
               c.challenge AS challenge,
               c.total_duration AS total_duration,
               collect({pose_name: p.id, order: rel.order, duration_seconds: rel.duration_seconds}) AS sequence
        """
        with self.neo4j_driver.session() as session:
            results = session.run(query, course_names=course_names)
            courses = []
            for record in results:
                seq = [PoseInSequence(**pose_dict) for pose_dict in record["sequence"]]
                courses.append(
                    CourseCandidate(
                        course_name=record["name"],
                        description=record["description"],
                        challenge=record["challenge"],
                        total_duration=record["total_duration"],
                        sequence=seq,
                    )
                )
            return courses

    def run(self, user_query: str, max_retries: int = 2):
        """
        Executes the full recommendation and validation pipeline.
        """
        # --- Phase 1: Try to find an existing course ---
        logging.info("--- Phase 1: Searching for existing courses ---")
        # Use Course Finder ADK agent
        try:
            resp = requests.post(
                self.course_finder_url,
                json={"user_query": user_query},
                timeout=60,
            )
            resp.raise_for_status()
            candidate_names = resp.json().get("courses", [])
        except Exception as e:
            logging.error(f"Error calling course finder agent: {e}")
            candidate_names = []

        courses = self._get_course_details(candidate_names)

        for course in courses:
            logging.info(f"\nValidating candidate course: '{course.course_name}'")
            original_sequence = [p.pose_name for p in course.sequence]
            
            validated_sequence = self._validate_sequence(original_sequence, user_query)
            
            if validated_sequence:
                print("\n🎉 Found an acceptable existing course!")
                print(f"Course Name: {course.course_name}")
                print("Validated Pose Sequence:")
                for i, pose in enumerate(validated_sequence, 1):
                    print(f"  {i}. {pose}")
                return

        # --- Phase 2: Fallback to composing a new course ---
        logging.info("\n--- Phase 2: No suitable existing course found. Composing a new one. ---")
        for i in range(max_retries):
            logging.info(f"Attempt {i + 1} of {max_retries}...")
            try:
                resp = requests.post(
                    self.category_url,
                    json={"user_query": user_query},
                    timeout=120,
                )
                resp.raise_for_status()
                sequence = resp.json().get("sequence", [])
            except Exception as e:
                logging.error(f"Error calling category recommender agent: {e}")
                sequence = []

            if not sequence:
                logging.warning("Category recommender failed to create a sequence. Retrying...")
                continue

            validated_sequence = self._validate_sequence(sequence, user_query)

            if validated_sequence:
                print("\n🎉 Successfully composed and validated a new course!")
                print("Composed Pose Sequence:")
                for i, pose in enumerate(validated_sequence, 1):
                    print(f"  {i}. {pose}")
                return

        # --- Phase 3: Failure ---
        print("\n😞 Sorry, after multiple attempts, we could not create a suitable yoga course for your query.")

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()

def main():
    parser = argparse.ArgumentParser(description="Yoga Application Runner")
    parser.add_argument(
        "--query",
        type=str,
        default="I need a 30-minute session for strength, but I have a weak neck and can't do headstands.",
        help="The user's natural language query.",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "deepseek"],
        default="deepseek",
        help="Specify which model API to use.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # ---------- Launch pose checker service ----------
    processes: list[subprocess.Popen] = []
    runner = None
    try:
        pose_cmd = [
            "python",
            "-m",
            "services.pose_checker.server",
            "--api",
            args.api,
            "--port",
            "0",
            "--host",
            "127.0.0.1",
        ]
        pose_proc = subprocess.Popen(pose_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        processes.append(pose_proc)

        def _wait_address(proc, tag):
            addr = None
            for ln in iter(proc.stdout.readline, ''):
                logging.info(f"[{tag}]: {ln.strip()}")
                m = re.search(r"Uvicorn running on (http://[0-9\.:]+)", ln)
                if m:
                    addr = m.group(1)
                    break
                if proc.poll() is not None:
                    raise RuntimeError(f"{tag} terminated before startup.")
            if addr is None:
                raise RuntimeError(f"Could not determine address for {tag}.")
            return addr

        pose_base = _wait_address(pose_proc, "PoseChecker")

        # ---------- Discover agent cards ----------
        cards = discover_agent_cards()
        required_actions = {
            "find_courses": None,
            "compose_course": None,
        }
        for act in required_actions:
            if act not in cards:
                raise RuntimeError(f"Required action '{act}' not found in any agent card.")

        # Start each agent server and build full endpoint URLs
        action_urls = {}
        for act_id, info in required_actions.items():
            card_info = cards[act_id]
            proc, base_url = start_agent_server(card_info["agent_dir"])
            processes.append(proc)
            endpoint = card_info["endpoint"]
            action_urls[act_id] = f"{base_url}{endpoint}"

        course_finder_url = action_urls["find_courses"]
        category_url = action_urls["compose_course"]

        # ---------- Run main application pipeline ----------
        runner = YogaApplicationRunner(
            pose_api_base=pose_base,
            course_finder_url=course_finder_url,
            category_url=category_url,
        )
        runner.run(user_query=args.query)

    except Exception as e:
        logging.error(f"An error occurred in the main runner: {e}")
    finally:
        if runner:
            runner.close()
        for p in processes:
            p.terminate()
            p.wait()
        logging.info("All subprocesses have been shut down.")

if __name__ == "__main__":
    main()
