import requests
import json
import os 

URL = "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/wikipedia/wiki_lookup.json"


def get_wikipedia_data():
	data = ''
	solditems = requests.get(URL) # (your url)
	data = solditems.json()
	with open('wikidata.json', 'w') as f:
	    json.dump(data, f)


# with open('../../data/wikidata.json', 'r') as f:
# 	data = json.load(f)



# for i, k in enumerate(data):
# 	if i > 10:
# 		break;
# 	print(k)
# 	print(data[k])

# print(data['Cytokinin'].keys())
# print(data['Cytokinin']['text'])