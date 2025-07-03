import os
import json
import argparse
import logging
from openai import OpenAI
from neo4j import GraphDatabase
from services.vector_store import get_client, embed

# Configuration - Load from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
# Qdrant persistence handled by services.vector_store
PROMPT_FILE_PATH = "/home/xli/NAS/home/bin/yoga-info-processing/get_user_query_key_info.prompt"


class CourseFinder:
    """
    A class to find yoga course candidates based on a user query using a RAG system.
    It combines semantic search on ChromaDB with graph lookups in Neo4j and LLM-based filtering.
    """

    def __init__(self, api_type: str):
        """
        Initializes the CourseFinder with API, Neo4j, and ChromaDB clients.

        Args:
            api_type (str): The model API to use ('openai' or 'deepseek').
        """
        self.api_client = None
        self.api_model = ""
        self._init_api_client(api_type)

        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # Embedded Qdrant client
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
        with open(PROMPT_FILE_PATH, 'r') as f:
            prompt_template = f.read().format(query=user_query)

        response = self.api_client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _search_courses_by_keywords(self, keywords: list, k: int = 5) -> list:
        """Semantic search in Qdrant for courses matching *keywords*."""
        course_names: set[str] = set()
        for kw in keywords:
            vector = embed([kw])[0]
            resp = self.qclient.query_points(
                collection_name="yoga_course",
                query=vector,
                limit=k,  # ask for more to improve recall
                with_payload=True,
            )
            course_names.update(p.payload["course"] for p in resp.points)
        return list(course_names)

    def _get_course_descriptions(self, course_names: list) -> dict:
        """Retrieves course descriptions from Neo4j."""
        with self.neo4j_driver.session() as session:
            result = session.run(
                "UNWIND $course_names AS name "
                "MATCH (c:Course {id: name}) "
                "RETURN c.id AS name, c.description AS description",
                course_names=course_names
            )
            return {record["name"]: record["description"] for record in result}

    def _filter_courses_by_llm(self, course_descriptions: dict, user_query: str) -> list:
        """Filters courses using LLM verification."""
        yes_courses = []
        na_courses = []

        for name, description in course_descriptions.items():
            prompt = (
                f"Please answer if the Yoga course matches the user's query for a training. "
                f"Course description: {description}\n"
                f"User query: {user_query}\n\n"
                "Answer only one of the three words: yes, no, or n/a"
            )

            response = self.api_client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=5
            )

            answer = response.choices[0].message.content.strip().lower()
            if answer == "yes":
                yes_courses.append(name)
            elif answer == "n/a":
                na_courses.append(name)

        return yes_courses if yes_courses else na_courses

    def find_candidates(self, user_query: str) -> list:
        """
        Main pipeline to get filtered course candidates.

        Args:
            user_query (str): The user's natural language query.

        Returns:
            list: A list of recommended course names.
        """
        # Step 1: Extract structured info from query
        query_info = self._extract_query_info(user_query)
        print(f"User's query info: {query_info}")

        # Step 2: Semantic search for candidates
        candidate_courses = set()

        # Search by objectives
        if query_info.get("objective"):
            candidate_courses.update(
                self._search_courses_by_keywords(
                    query_info["objective"],
                    k=3
                )
            )

        logging.info(f"Candidate courses by objectives: {candidate_courses}")

        # Search by body parts
        if query_info.get("physical body parts to train"):
            candidate_courses.update(
                self._search_courses_by_keywords(
                    query_info["physical body parts to train"],
                    k=3
                )
            )

        logging.info(f"Candidate courses by body parts: {candidate_courses}")
        
        if not candidate_courses:
            return []

        # Step 3: Filter courses using LLM verification
        course_descriptions = self._get_course_descriptions(list(candidate_courses))
        return self._filter_courses_by_llm(course_descriptions, user_query)

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get yoga course candidates based on a user query."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Please suggest a 30-min yoga sequence avoiding pressuring the wrist",
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

    finder = None
    try:
        finder = CourseFinder(api_type=args.api)
        candidates = finder.find_candidates(user_query=args.query)
        print(f"Finding course candidates for query: '{args.query}' using API: {args.api}")
        print(f"Candidate courses: {candidates}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if finder:
            finder.close()
