# wopper/test/wopper_test.py
# Basic smoke tests for the Wikidata interface and placeholders for future tests.

from pathlib import Path
import sys
import os

# Try to import optional dependencies. If unavailable, create minimal stubs
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - if python-dotenv not installed
    def load_dotenv(*args, **kwargs):
        return False

# Ensure the project root is on the Python path when executing this
# file directly as ``python3 test/wopper_test.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logger import get_logger

log = get_logger(__name__)
log.debug("Starting wopper_test.py")

# Import the Wikidata interface directly to avoid pulling in optional
# ChatGPT dependencies. Gracefully handle missing packages.
try:
    from interface.wikidata_interface import WikidataInterface  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    log.warning("WikidataInterface unavailable: %s", exc)
    WikidataInterface = None


def test_wikidata_interface():
    if WikidataInterface is None:
        log.warning("Skipping WikidataInterface test due to missing dependencies")
        return

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
    load_dotenv(os.path.expanduser("~/.env"))

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

