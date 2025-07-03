# sanity_check_qdrant.py
"""
Checks that the embedded Qdrant database contains non-empty
`yoga_pose`, `yoga_course`, and `yoga_category` collections and that
vectors can be queried and retrieved with payloads.
"""

from services.vector_store import get_client, embed, shutdown_server
import pathlib, os

COLLECTIONS = ["yoga_course", "yoga_category", "yoga_pose"]

def main() -> None:
    qc = get_client()
    for col in COLLECTIONS:
        info = qc.get_collection(col)
        print(f"\n=== {col} ===")
        print("status:", info.status, "| points:", info.points_count)

        if info.points_count == 0:
            print("⚠️  Collection is EMPTY — rebuild needed")
            continue

        # Pick a simple query vector (embedding of collection name)
        vector = embed([col.replace("yoga_", "").replace("_", " ")])[0]
        resp = qc.query_points(
            collection_name=col,
            query=vector,
            with_payload=True,
            limit=3,
        )
        print("sample hits:")
        for p in resp.points:
            print(" • id:", p.id, "| payload:", p.payload)

    qc.close()
    shutdown_server()

if __name__ == "__main__":
    main()
