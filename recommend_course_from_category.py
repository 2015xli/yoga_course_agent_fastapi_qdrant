import os
import json
import argparse
from openai import OpenAI
from neo4j import GraphDatabase
from services.vector_store import get_client, embed

# Configuration - Load from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
# Qdrant persistence handled by services.vector_store
PROMPT_FILE_PATH = "/home/xli/NAS/home/bin/yoga-info-processing/get_user_query_key_info.prompt"

class CategoryCourseRecommender:
    """
    Recommends a yoga course by finding poses from relevant categories based on user objectives.
    """

    def __init__(self, api_type: str):
        """
        Initializes the recommender with API, Neo4j, and ChromaDB clients.

        Args:
            api_type (str): The model API to use ('openai' or 'deepseek').
        """
        self.api_client = None
        self.api_model = ""
        self._init_api_client(api_type)

        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        self.qclient = get_client()

    def _init_api_client(self, api_type: str):
        """Initializes the LLM API client."""
        if api_type == 'deepseek':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY")
            self.api_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
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

    def _find_similar_categories(self, objectives: list, k: int = 2) -> list:
        """Find similar yoga categories via Qdrant."""
        if not objectives:
            return []
        category_names: set[str] = set()
        for obj in objectives:
            vec = embed([obj])[0]
            resp = self.qclient.query_points(
                collection_name="yoga_category",
                query=vec,
                limit=k,
                with_payload=True,
            )
            category_names.update(p.payload["category"] for p in resp.points)
        return list(category_names)

    def _get_random_pose_for_category(self, tx, category_name: str) -> str | None:
        """Gets a random pose name for a given category from Neo4j."""
        result = tx.run(
            """
            MATCH (c:Category {id: $category_name})<-[:IN_CATEGORY]-(p:Pose)
            RETURN p.id AS pose_name
            ORDER BY rand()
            LIMIT 1
            """,
            category_name=category_name
        )
        record = result.single()
        return record["pose_name"] if record else None

    def _find_related_poses(self, tx, pose_name: str) -> dict:
        """Finds preceding and succeeding poses for a given pose."""
        preceding_query = """
        MATCH (preceding:Pose)<-[:BUILD_UP]-(current:Pose {id: $pose_name})
        RETURN preceding.id AS pose_name
        LIMIT 1
        """
        succeeding_query = """
        MATCH (current:Pose {id: $pose_name})-[:BALANCE_OUT|UNWIND]->(succeeding:Pose)
        RETURN succeeding.id AS pose_name
        ORDER BY rand()
        LIMIT 1
        """
        
        preceding_result = tx.run(preceding_query, pose_name=pose_name).single()
        succeeding_result = tx.run(succeeding_query, pose_name=pose_name).single()
        result = {
            "preceding": preceding_result["pose_name"] if preceding_result else None,
            "succeeding": succeeding_result["pose_name"] if succeeding_result else None,
        }

        return result


    def recommend_course(self, user_query: str) -> list:
        """
        Main pipeline to generate a course from category-based pose selection.

        Args:
            user_query (str): The user's natural language query.

        Returns:
            list: A list of pose names for the recommended course.
        """
        # Step 1: Extract user objective
        query_info = self._extract_query_info(user_query)
        objectives = query_info.get("objective", [])
        if not objectives:
            print("No objective found in the query.")
            return []

        # Step 2: Find similar categories
        similar_categories = self._find_similar_categories(objectives, k=2)
        if not similar_categories:
            print("No similar categories found.")
            return []
        
        print(f"Found similar categories: {similar_categories}")

        # Step 3: Build sequence for each category
        final_sequence = []
        with self.neo4j_driver.session() as session:
            for category in similar_categories:
                current_pose = session.execute_read(self._get_random_pose_for_category, category)
                if not current_pose:
                    print(f"No pose found for category: {category}")
                    continue
                
                related_poses = session.execute_read(self._find_related_poses, current_pose)
                
                mini_sequence = []
                if related_poses.get("preceding"):
                    mini_sequence.append(related_poses["preceding"])
                
                mini_sequence.append(current_pose)
                
                if related_poses.get("succeeding"):
                    mini_sequence.append(related_poses["succeeding"])
                
                print(f"Generated sequence for category '{category}': {mini_sequence}")
                final_sequence.extend(mini_sequence)
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in final_sequence if not (x in seen or seen.add(x))]

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recommend a yoga course from categories based on a user query."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="I want to find my bodily balance and calm a scattered mind",
        help="The input string from the user as a query.",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "deepseek"],
        default="deepseek",
        help="Specify which model API to use (openai or deepseek).",
    )
    args = parser.parse_args()

    recommender = None
    try:
        recommender = CategoryCourseRecommender(api_type=args.api)
        recommended_course = recommender.recommend_course(user_query=args.query)
        print(f"\n--- Recommended Course for query: '{args.query}' ---")
        if recommended_course:
            for i, pose in enumerate(recommended_course, 1):
                print(f"{i}. {pose}")
        else:
            print("Could not generate a recommendation.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if recommender:
            recommender.close()
