import os, json
from pathlib import Path
import logging
from neo4j import GraphDatabase
from services.vector_store import get_client, embed, recreate_collection, str2uuid, shutdown_server
import shutil
from qdrant_client import models

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
#INPUT_DATA_DIR = "/home/xli/NAS/home/bin/yoga_course_agent_fastapi_qdrant/data"
INPUT_DATA_DIR = Path(__file__).resolve().parent / "data"
# Qdrant persistence handled by services.vector_store
POSE_JSON = f"{INPUT_DATA_DIR}/array_pose.json"
ATTRIBUTE_JSON = f"{INPUT_DATA_DIR}/array_attribute.json"
CATEGORY_JSON = f"{INPUT_DATA_DIR}/array_category.json"
CHALLENGE_JSON = f"{INPUT_DATA_DIR}/array_challenge.json"
COURSE_JSON = f"{INPUT_DATA_DIR}/array_course.json"
QDRANT_COLLECTION_POSE = "yoga_pose"
QDRANT_COLLECTION_COURSE = "yoga_course"
QDRANT_COLLECTION_CATEGORY = "yoga_category"

def delete_chroma_collection(chroma_client, collection_name: str):
    existing = [col.name for col in chroma_client.list_collections()]
    if collection_name in existing:
        chroma_client.delete_collection(collection_name)
        print(f"✅ Deleted collection: {collection_name}")
    else:
        print(f"⚠️ Collection not found: {collection_name} - skipping deletion")

def delete_neo4j_database(driver):
    """Clear existing data in Neo4j and ChromaDB"""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path) as f:
        return json.load(f)

def create_neo4j_nodes(tx, node_type, items, id_field):
    """Create nodes in Neo4j"""
    query = f"""
    UNWIND $items AS item
    MERGE (n:{node_type} {{id: item.{id_field}}})
    SET n += apoc.map.removeKeys(item, ['{id_field}'])
    """
    tx.run(query, items=items)

def link_pose_to_references(tx):
    """Create relationships between poses and reference nodes"""
    link_query = """
    MATCH (p:Pose), (ref:{reference_type} {{id: p.{reference_field}}})
    MERGE (p)-[:{relationship}]->(ref)
    """
    references = [
        ("Attribute", "attribute", "HAS_ATTRIBUTE"),
        ("Category", "category", "IN_CATEGORY"),
        ("Challenge", "challenge", "HAS_CHALLENGE")
    ]
    
    for ref_type, field, rel in references:
        tx.run(link_query.format(
            reference_type=ref_type,
            reference_field=field,
            relationship=rel
        ))

def create_pose_relationships(tx, pose):
    """Create relationships between poses"""
    relationships = {
        "BUILD_UP": pose.get("build_up", []),
        "MOVE_FORWARD": pose.get("move_forward", []),
        "BALANCE_OUT": pose.get("balance_out", []),
        "UNWIND": pose.get("unwind", [])
    }
    
    for rel_type, targets in relationships.items():
        for target in targets:
            tx.run("""
            MATCH (source:Pose {id: $source_name})
            MATCH (target:Pose {id: $target_name})
            MERGE (source)-[:%s]->(target)
            """ % rel_type,
            source_name=pose["name"],
            target_name=target)

# Update the create_course_nodes function
def create_course_nodes(tx, courses):
    """Create course nodes and relationships with support for repeated poses"""
    for course in courses:
        # Create course node
        tx.run("""
        MERGE (c:Course {id: $name})
        SET c += {
            challenge: $challenge,
            description: $description,
            total_duration: $total_duration
        }
        """, 
        name=course["name"],
        challenge=course["challenge"],
        description=course["description"],
        total_duration=course["total_duration"])
        
        # Create sequence relationships with unique identifiers
        for i, step in enumerate(course["sequence"]):
            # Generate unique relationship ID
            rel_id = f"{course['name']}_{step['pose']}_{i}"
            
            tx.run("""
            MATCH (c:Course {id: $course_name})
            MATCH (p:Pose {id: $pose_name})
            MERGE (c)-[rel:INCLUDES_POSE {
                id: $rel_id,
                order: $order
            }]->(p)
            SET rel += {
                duration_seconds: $duration_seconds,
                repeat_times: $repeat_times,
                transition_note: $transition_notes,
                action_note: $action_note
            }
            """,
            course_name=course["name"],
            pose_name=step["pose"],
            rel_id=rel_id,
            order=i+1,
            duration_seconds=step["duration_seconds"],
            repeat_times=step["repeat_times"],
            transition_notes=step["transition_notes"],
            action_note=step["action_note"])
        
        # Link course to challenge
        tx.run("""
        MATCH (c:Course {id: $name})
        MATCH (ch:Challenge {level: $challenge})
        MERGE (c)-[:HAS_CHALLENGE]->(ch)
        """,
        name=course["name"],
        challenge=course["challenge"])

def build_knowledge_graph(driver):
    """Main function to build the knowledge graph"""   
    # Load data
    pose_data = load_json_data(POSE_JSON)["pose"]
    attributes = load_json_data(ATTRIBUTE_JSON)["attribute"]
    categories = load_json_data(CATEGORY_JSON)["category"]
    challenges = load_json_data(CHALLENGE_JSON)["challenge"]
    courses = load_json_data(COURSE_JSON)["course"]
    
    with driver.session() as session:
        # Create reference nodes
        session.execute_write(create_neo4j_nodes, "Attribute", attributes, "name")
        session.execute_write(create_neo4j_nodes, "Category", categories, "name")
        session.execute_write(create_neo4j_nodes, "Challenge", challenges, "level")
        
        # Create pose nodes
        session.execute_write(create_neo4j_nodes, "Pose", pose_data, "name")

        # update the challenge property to long type to match Challenge.id
        session.execute_write(
            lambda tx: tx.run("""
                MATCH (p:Pose)
                SET p.challenge = toInteger(p.challenge)
                """))
        
        # update Attribute id to capital letter to match Pose.attribute
        session.execute_write(
            lambda tx: tx.run("""
                MATCH (a:Attribute)
                SET a.id = apoc.text.capitalize(a.id)
            """))
        
        # Link poses to references
        session.execute_write(link_pose_to_references)
        
        # Create inter-pose relationships
        for pose in pose_data:
            session.execute_write(create_pose_relationships, pose)

        # create courses
        session.execute_write(create_course_nodes, courses)

    driver.close()
    print("Knowledge graph built successfully!")
    print(f"Created {len(pose_data)} pose nodes")

def check_neo4j_dbms_connection(driver):
    try:
        driver.verify_connectivity()
        print("✅ Connection established!")
        # Optionally perform a quick test query:
        with driver.session() as session:
            result = session.run("RETURN 1 AS result").single()
            print("Test query result:", result["result"])
    except Exception as e:
        print("❌ Connection failed:", e)
    finally:
        #driver.close()
        print("Checked!")

# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def add_documents_to_qdrant(client, collection_name: str, ids: list[str], texts: list[str], payloads: list[dict]):
    """Utility to recreate collection and upsert documents with embeddings.
    Converts human-readable ids to deterministic UUID strings required by Qdrant.
    """
    recreate_collection(client, collection_name, dim=len(embed(["test"])[0]))
    vectors = embed(texts)
    uuid_ids = [str2uuid(i) for i in ids]
    client.upsert(
        collection_name=collection_name,
        wait=True,  # ensure data is flushed before closing client
        points=models.Batch(ids=uuid_ids, vectors=vectors, payloads=payloads),
    )
    assert uuid_ids and vectors, "Empty lists!"
    print("✅ Inserted", len(uuid_ids), "documents into", collection_name)
    
    info = qclient.get_collection(collection_name)
    print("points:", info.points_count)

    pts, _ = qclient.scroll(collection_name, limit=3, with_payload=True, with_vectors=False)
    print("example points:", pts)

    # Force persistence to disk so subsequent processes can see the data
    if hasattr(client._client, "flush"):
        try:
            client._client.flush()
        except Exception as e:
            logging.warning("Qdrant flush failed: %s", e)

if __name__ == "__main__":
    # Build/refresh vector collections in embedded Qdrant
    qclient = get_client()

    # ---- Build pose collection (optional for current features) ----
    pose_data = load_json_data(POSE_JSON)["pose"]
    pose_texts, pose_ids, pose_payloads = [], [], []
    for pose in pose_data:
        # Combine fields for semantic search
        doc = "\n".join([
            "Introduction: " + pose.get("introduction", ""),
            "Steps: " + " ".join(pose.get("steps", [])),
            "Effects: " + pose.get("effects", ""),
        ])
        pose_ids.append(pose["name"])
        pose_texts.append(doc)
        pose_payloads.append({"pose": pose["name"]})

    add_documents_to_qdrant(qclient, QDRANT_COLLECTION_POSE, pose_ids, pose_texts, pose_payloads)
    
    # ---- Build course collection ----
    course_data = load_json_data(COURSE_JSON)["course"]
    course_texts, course_ids, course_payloads = [], [], []
    for course in course_data:
        doc = (
            f"Course: {course['name']}\n"
            f"Challenge Level: {course['challenge']}\n"
            f"Total Duration: {course['total_duration']}\n"
            f"Description: {course['description']}\n"
        )
        course_ids.append(course["name"])
        course_texts.append(doc)
        course_payloads.append({"course": course["name"]})

    add_documents_to_qdrant(qclient, QDRANT_COLLECTION_COURSE, course_ids, course_texts, course_payloads)

    # ---- Build category collection ----
    category_data = load_json_data(CATEGORY_JSON)["category"]
    cat_texts, cat_ids, cat_payloads = [], [], []
    for cat in category_data:
        doc = (
            f"Category: {cat['name']}\n"
            f"Introduction: {cat.get('introduction', '')}\n"
            f"Guidelines: {' '.join(cat.get('guidelines', []))}\n"
        )
        cat_ids.append(cat["name"])
        cat_texts.append(doc)
        cat_payloads.append({"category": cat["name"]})    

    add_documents_to_qdrant(qclient, QDRANT_COLLECTION_CATEGORY, cat_ids, cat_texts, cat_payloads)

    print("✅ Qdrant collections built/updated.")

    # Close the embedded Qdrant client to release the file lock before exiting.
    qclient.close()
    shutdown_server()

    # ---- Build Knowledge Graph ----
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    check_neo4j_dbms_connection(driver)
    delete_neo4j_database(driver)
    build_knowledge_graph(driver)
