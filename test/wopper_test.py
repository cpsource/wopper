# wopper/test/wopper_test.py

import sys
import os
from dotenv import load_dotenv

# Ensure interface directory is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'interface')))

from wikidata_interface import WikidataInterface


def test_wikidata_interface():
    print("🔍 Testing: Wikidata Interface")

    wikidata = WikidataInterface()
    subclasses = wikidata.get_subclasses("Q1292119")  # Q1292119 = grocery store

    if subclasses:
        print("✅ Received subclass results:")
        for label in subclasses[:5]:
            print(f"   • {label}")
    else:
        print("❌ No results returned from SPARQL query.")

def main():
    print("🚀 WOPPER Top-Level Test Runner")
    print("=" * 40)

    test_wikidata_interface()

    # Placeholder for future system-wide tests:
    # - concept_inference()
    # - auto_model_spawn()
    # - db_connection()
    # - self-correction evaluation()

    print("\n✅ WOPPER tests complete.")

if __name__ == "__main__":
    main()

