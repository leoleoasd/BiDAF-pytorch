import sys
import json
import random
import os

ratio = 0.9
filename = sys.argv[1]

with open(filename) as f:
    data = json.load(f)

paragraphs = data['data'][0]['paragraphs']
random.shuffle(paragraphs)

with open(os.path.splitext(filename)[0] + "-train.json", "w") as f:
    json.dump({
        'data': [
            {
                'title': data['data'][0]['title'] + "-train",
                'paragraphs': paragraphs[:int(len(paragraphs) * ratio)]
            }
        ],
        'version': '1.0'
    }, f, indent=2)

with open(os.path.splitext(filename)[0]  + "-dev.json", "w") as f:
    json.dump({
        'data': [
            {
                'title': data['data'][0]['title'] + "-dev",
                'paragraphs': paragraphs[int(len(paragraphs) * ratio):]
            }
        ],
        'version': '1.0'
    }, f, indent=2)
