
def fetch_entities(query,url,headers):
    l = list()
    response = requests.get(url, params={"query": query, "format": "json"}, headers=headers)

    if response.status_code == 200:
        data = response.json()
        for x in data["results"]["bindings"]:
            l.append(x["item"]["value"].split('/')[-1])
    else:
        print(f"Error: {response.status_code}")

    return l


def find_claims(a_list):
        wikidata = 'https://www.wikidata.org/w/api.php'
        wd_params =  {'format':'json','action':'wbgetentities','ids':'{}'.format('|'.join(a_list))}
        req = requests.get(wikidata,params=wd_params,headers=headers)
        counted_rels = list()
        
        data = req.json()['entities']

        return data

jsn = json.load(open('categories_mapping.json'))
mydict = dict()
for item in list(jsn)[11:]:
    mydict[item] = {}
    f = open(f'raw_results/{item}.jsonl',mode='w')

    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "MyPythonApp/1.0 (https://example.com; myemail@example.com)"
    }



    l = fetch_entities(query=jsn[item]['query'],url=url,headers=headers)
    l = [l[i:i + 50] for i in range(0, len(l), 50)]

    for chunk in tqdm(l):
        data = find_claims(chunk)
        for el in data:
            try:
                f.write(json.dumps(data[el])+'\n')
            except:
                continue

    f.close()