# wopper/test/wopper_test.py
# Basic smoke tests for the Wikidata interface and placeholders for future tests.

import sys
import os
from dotenv import load_dotenv
from logger import get_logger

log = get_logger(__name__)
log.debug("Starting wopper_test.py")

# Ensure interface directory is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'interface')))

from wikidata_interface import WikidataInterface


def test_wikidata_interface():
    log.info("🔍 Testing: Wikidata Interface")

    wikidata = WikidataInterface()
    subclasses = wikidata.get_subclasses("Q1292119")  # Q1292119 = grocery store

    if subclasses:
        log.info("✅ Received subclass results:")
        for label in subclasses[:5]:
            log.info("   • %s", label)
    else:
        log.error("❌ No results returned from SPARQL query.")

def main():
    log.info("🚀 WOPPER Top-Level Test Runner")
    log.info("=" * 40)

    test_wikidata_interface()

    # Placeholder for future system-wide tests:
    # - concept_inference()
    # - auto_model_spawn()
    # - db_connection()
    # - self-correction evaluation()

    log.info("\n✅ WOPPER tests complete.")

if __name__ == "__main__":
    main()

