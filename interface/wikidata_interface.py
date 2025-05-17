# Helper for running SPARQL queries against Wikidata.
import os
from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON
from utils.logger import get_logger

log = get_logger(__name__)
log.debug("Starting wikidata_interface.py")

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
            results = self.sparql.query().convert()
            return results
        except Exception as e:
            log.error("SPARQL query failed: %s", e)
            return None

    def get_subclasses(self, concept_id):
        """
        Returns a list of (QID, label) tuples for valid subclasses of a given Wikidata concept,
        filtered to only return results that are declared instances of 'class'.

        Args:
            concept_id (str): A Wikidata QID (e.g., 'Q2134413' for shop)

        Returns:
            list of tuples: [(qid, label), ...]
        """
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
            ?item wdt:P279 wd:{concept_id}.          # Direct subclass
            ?item wdt:P31 wd:Q5127848.               # Must be an instance of 'class'
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en".
            }}
        }}
        """
        results = self._run_query(query)
        if results:
            return [
                (res['item']['value'].split('/')[-1], res['itemLabel']['value'])
                for res in results['results']['bindings']
            ]
        return []

# ------------------------------------------
# 🧪 Local test
# ------------------------------------------
if __name__ == "__main__":
    wikidata = WikidataInterface()
    concept_id = "Q2134413"  # 'shop'
    subclasses = wikidata.get_subclasses(concept_id)

    log.info(f"Subclasses of {concept_id} (shop):")
    for qid, label in subclasses[:15]:
        log.info(" - %s (%s)", label, qid)

