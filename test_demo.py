import json

with open('demo.txt', "r", encoding="utf-8") as f:
    data = json.load(f)

print(data)

