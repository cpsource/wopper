from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataInterface:
    def __init__(self, endpoint="https://query.wikidata.org/sparql"):
        self.endpoint = endpoint
        self.sparql = SPARQLWrapper(endpoint)

    def _run_query(self, query):
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        try:
            results = self.sparql.query().convert()
            return results
        except Exception as e:
            print("SPARQL query failed:", e)
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

    print(f"Subclasses of {concept_id} (shop):")
    for qid, label in subclasses[:15]:
        print(f" - {label} ({qid})")

