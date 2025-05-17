# Legacy wrapper around SPARQLWrapper to query Wikidata.
import os
from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON
from utils.logger import get_logger

log = get_logger(__name__)
log.debug("Starting wididatainterface.py")

class WikidataInterface:
    def __init__(self, endpoint="https://query.wikidata.org/sparql"):
        load_dotenv(os.path.expanduser("~/.env"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in ~/.env")
        self.endpoint = endpoint
        self.sparql = SPARQLWrapper(endpoint)
    
    def _run_query(self, query):
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        try:
            return self.sparql.query().convert()
        except Exception as e:
            log.error("SPARQL query failed: %s", e)
            return None
    
    def get_subclasses(self, concept_id):
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
            ?item wdt:P279* wd:{concept_id}.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        results = self._run_query(query)
        if results:
            return [res['itemLabel']['value'] for res in results['results']['bindings']]
        return []


# --------------------------
# 🧪 Local test function
# --------------------------
def main():
    """Run a simple subclass query as a smoke test."""
    wikidata = WikidataInterface()
    subclasses = wikidata.get_subclasses("Q1292119")  # grocery store
    if subclasses:
        log.info("Found subclasses:")
        for label in subclasses[:5]:
            log.info(" - %s", label)
    else:
        log.warning("No results returned from SPARQL query.")


if __name__ == "__main__":
    main()

