import os
from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON

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
            print("SPARQL query failed:", e)
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

