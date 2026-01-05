from SPARQLWrapper import SPARQLWrapper, JSON
import requests

class PopularityRetriever:

    def __init__(self):
        self.header = {
        "User-Agent": ""
    }
        self.languages = ["en", "it","es"]
        self.pageviews_api = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
        self.pagelength = "https://en.wikipedia.org/api/rest_v1/page/summary"

    
    def get_pageviews(self,article, project, start=20191101,end=20251101):
        

        url = f"{self.pageviews_api}/{project}/all-access/all-agents/{article}/daily/{start}/{end}"

        
        response = requests.get(url, headers=self.header)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code}")

        data = response.json()
        total_views = sum(item['views'] for item in data['items'])

        return total_views
    
    

    def get_page_text_length(self,article,project):
        url = f"https://{project}.wikipedia.org/w/api.php"
        params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,   # plain text (no HTML)
        "titles": article,
        "format": "json"
        }


        response = requests.get(url=url, params=params, headers=self.header)

        if response.status_code != 200:
            raise Exception(f"Error fetching page: {response.status_code}")

        data = response.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))  # Get first (and usually only) page
        text = page.get("extract", "")

        return len(text)

        

        response = requests.get(url, headers=self.header)
        if response.status_code != 200:
            raise Exception(f"Error fetching page: {response.status_code}")

        data = response.json()
        text = data.get("extract", "")  # Extract is the plain-text summary
        return len(text), text

    def get_wikipedia_title(self,dbentity):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        query = f"""
        SELECT ?wikipediaPage WHERE {{
            dbr:{dbentity} foaf:isPrimaryTopicOf ?wikipediaPage .
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        pages = [result['wikipediaPage']['value'] for result in results['results']['bindings'] if 'en.wikipedia.org' in result['wikipediaPage']['value']]
        
        title = pages[0].split('/')[-1] if pages else None
        
        return title
    
    def get_wikimedia_item_from_title(self,title, language="en"):
        url = f"https://{language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "pageprops",
            "format": "json"
        }
        
        response = requests.get(url, params=params, headers=self.header)
        response.raise_for_status()  # good practice to catch request errors
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "pageprops" in page_data and "wikibase_item" in page_data["pageprops"]:
                return page_data["pageprops"]["wikibase_item"]
        return None
    
    def get_wikipedia_pages(self,wdid):
        url = f"https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": wdid,
            "props": "sitelinks",
            "format": "json"
        }
        response = requests.get(url, params=params, headers=self.header)
        response.raise_for_status()
        data = response.json()
        
        sitelinks = data.get("entities", {}).get(wdid, {}).get("sitelinks", {})
        wikipedia_pages = {}
        
        for lang in self.languages:
            try:
                
                site_key = f"{lang}wiki"
                if site_key in sitelinks:
                    wikipedia_pages[lang] = sitelinks[site_key]["title"]
            except Exception as e:
                print(f"Error processing language {lang}: {e}")
        
        return wikipedia_pages