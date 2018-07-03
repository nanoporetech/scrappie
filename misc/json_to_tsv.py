import json
from sys import stdin


first_entry = json.loads(stdin.readline())
keys = list(first_entry.keys())
print('\t'.join(keys))
print('\t'.join( str(v) for v in first_entry.values()))

for line in  stdin.readlines():
    entry = json.loads(line)
    print('\t'.join( str(entry[k]) for k in keys ))
