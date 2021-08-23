import os
import json

MAP = {'걷거나 달리는 동작':0, '그루밍함':1, '걷거나 뜀 ':0, '걷거나 뛰는 동작':0, '그루밍하는 동작':1}

def getActionLabel(src):
    actionDict = {}

    for vid in sorted(os.listdir(src)):
        json_f = os.path.join(src, vid)
        if json_f[-4:] != "json":
            continue
        with open(json_f, "r") as f:
            data = json.load(f)
            act = data['metadata']['action']
            key = vid.split(".")[0]
            actionDict[key] = MAP[act]
    return actionDict

def saveActionLabel(actionDict, output):
    filename = os.path.join(output, "actionLabel.json")
    with open(filename, "w") as f:
        json.dump(actionDict, f)

src = '/home/butlely/Desktop/Dataset/aihub/label_7'
out = '/home/butlely/Desktop/Dataset/aihub/yolact_label'
act = getActionLabel(src)
saveActionLabel(act, out)









