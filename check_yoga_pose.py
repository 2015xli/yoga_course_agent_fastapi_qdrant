
import os
import json
import argparse
from openai import OpenAI
from neo4j import GraphDatabase

# Configuration - Load from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
PROMPT_FILE_PATH = "/home/xli/NAS/home/bin/yoga-info-processing/get_user_query_key_info.prompt"

class YogaPoseChecker:
    """
    A class to check if a yoga pose is suitable based on user-defined contraindications
    and find a replacement from the knowledge graph if it's not.
    """

    def __init__(self, api_type: str):
        """
        Initializes the YogaPoseChecker with API and Neo4j clients.

        Args:
            api_type (str): The model API to use ('openai' or 'deepseek').
        """
        self.api_client = None
        self.api_model = ""
        self._init_api_client(api_type)
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def _init_api_client(self, api_type: str):
        """Initializes the LLM API client."""
        if api_type == 'deepseek':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY")
            self.api_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            self.api_model = "deepseek-chat"
        elif api_type == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY")
            self.api_client = OpenAI(api_key=api_key)
            self.api_model = "gpt-3.5-turbo"
        else:
            raise ValueError("Unsupported API type")

    def _extract_query_info(self, user_query: str) -> dict:
        """Extracts structured info from user query using the LLM."""
        try:
            with open(PROMPT_FILE_PATH, 'r') as f:
                prompt_template = f.read().format(query=user_query)
        except FileNotFoundError:
            raise RuntimeError(f"Prompt file not found at {PROMPT_FILE_PATH}")

        response = self.api_client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _get_pose_caution(self, tx, pose_name: str) -> str:
        """Retrieves the caution string for a given pose from Neo4j."""
        result = tx.run("MATCH (p:Pose {id: $pose_name}) RETURN p.caution AS caution", pose_name=pose_name)
        record = result.single()
        return record["caution"] if record and record["caution"] else ""

    def _is_pose_unsuitable(self, pose_name: str, caution: str, poses_to_avoid: list, contraindications: list) -> bool:
        """
        Checks with the LLM if a pose is unsuitable.

        Returns:
            bool: True if the pose is unsuitable, False otherwise.
        """
        if not poses_to_avoid and not contraindications:
            return False

        prompt = (
            f"Here is a yoga pose '{pose_name}' and its practice caution: '{caution}'.\n"
            f"Check if the pose is similar to any pose in the list of poses to avoid: {poses_to_avoid}.\n"
            f"Also, check if practicing this pose has any contraindications listed here: {contraindications}.\n"
            f"Please answer only 'true' if it is unsuitable or 'false' if it is suitable, nothing else."
        )

        response = self.api_client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer == 'true'

    def _find_replacement_pose(self, tx, original_pose_name: str, poses_to_avoid: list, contraindications: list) -> str | None:
        """
        Finds a suitable replacement pose from the same category in Neo4j.
        """
        # Find other poses in the same category, excluding the original pose
        query = """
        MATCH (original:Pose {id: $original_pose_name})-[:IN_CATEGORY]->(cat:Category)<-[:IN_CATEGORY]-(replacement:Pose)
        WHERE original <> replacement
        RETURN replacement.id AS name, replacement.caution AS caution
        ORDER BY rand()
        """
        results = tx.run(query, original_pose_name=original_pose_name)
        
        for record in results:
            replacement_name = record["name"]
            replacement_caution = record["caution"] if record["caution"] else ""
            
            # Check if the replacement is suitable
            if not self._is_pose_unsuitable(replacement_name, replacement_caution, poses_to_avoid, contraindications):
                print(f"Found suitable replacement: {replacement_name}")
                return replacement_name
        
        print(f"Could not find a suitable replacement for {original_pose_name}")
        return None

    def check_and_replace_pose(self, pose_name: str, user_query: str) -> str | None:
        """
        Checks if a pose is suitable based on the user query. If not, finds and returns a replacement.
        If the pose is suitable, it returns the original pose name.
        If unsuitable and no replacement is found, returns None.
        """
        query_info = self._extract_query_info(user_query)
        poses_to_avoid = query_info.get("poses to avoid", [])
        contraindications = query_info.get("contraindications", [])

        # If there are no restrictions, the pose is suitable
        if not poses_to_avoid and not contraindications:
            return pose_name

        with self.neo4j_driver.session() as session:
            caution = session.execute_read(self._get_pose_caution, pose_name)
            
            is_unsuitable = self._is_pose_unsuitable(pose_name, caution, poses_to_avoid, contraindications)

            if is_unsuitable:
                print(f"Pose '{pose_name}' is unsuitable. Finding a replacement...")
                replacement = session.execute_read(self._find_replacement_pose, pose_name, poses_to_avoid, contraindications)
                return replacement
            else:
                print(f"Pose '{pose_name}' is suitable.")
                return pose_name

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if a yoga pose is suitable for a user and find a replacement if not."
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="headstand_pose",
        help="The name of the yoga pose to check.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Please suggest a yoga sequence that can improve my arm strength. My neck cannot do poses that requires headstand.",
        help="The user's query detailing their limitations.",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "deepseek"],
        default="deepseek",
        help="Specify which model API to use (openai or deepseek).",
    )
    args = parser.parse_args()

    checker = None
    try:
        checker = YogaPoseChecker(api_type=args.api)
        final_pose = checker.check_and_replace_pose(pose_name=args.pose, user_query=args.query)
        print(f"\n--- Pose Check Result ---")
        print(f"Original Pose: {args.pose}")
        print(f"User Query: '{args.query}'")
        print(f"Final Recommended Pose: {final_pose}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if checker:
            checker.close()
