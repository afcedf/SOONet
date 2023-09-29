import os
import re
import json
import csv

splits = ["train", "val", "test"]
root_dir = "data/mad/annotations"

for split in splits:
    with open(os.path.join(root_dir, "MAD_{}.json".format(split)), 'r') as f:
        raw_anns = json.load(f)


    annos = list()
    for qid, ann in raw_anns.items():
        vid = ann["movie"]
        duration = ann["movie_duration"]
        spos, epos = ann["ext_timestamps"]
        query = re.sub("\n", "", ann["sentence"])
        
        annos.append([str(qid), str(vid), str(duration), str(spos), str(epos), query])

    with open("data/mad/annotations/{}.txt".format(split), 'w') as f:
        for anno in annos:
            f.writelines(" | ".join(anno) + "\n")