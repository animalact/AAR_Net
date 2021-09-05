import os
import csv


def loadCsv(csvfile):
    data = {}
    with open(csvfile, "r") as f:
        reader = csv.DictReader(f)
        for item in reader:
            if not data.get(item["action_id"], None):
                data[item["action_id"]] = []
            data[item['action_id']].append(item)
    return data


def splitData(data, ratio=0.8):
    train = []
    test = []
    for action_id in data:
        action_data = data[action_id]
        count = len(action_data)
        tr_count = int(count * ratio)
        te_count = count - tr_count
        train.extend(action_data[:tr_count])
        test.extend(action_data[tr_count:])
    return train, test


def saveCsv(filename, data):
    filenames = list(data[0].keys())
    with open(filename, "w") as f:
        writer = csv.DictWriter(f, filenames)
        writer.writeheader()
        writer.writerows(data)


def run():
    for i in range(1,10):
        print(f"label_{i} started")
        data_folder = "/home/butlely/PycharmProjects/AAR_Net/data/"
        csvfilename = f"label_{i}.csv"
        csvfile = os.path.join(data_folder, csvfilename)
        data = loadCsv(csvfile)
        tr, ts = splitData(data)
        saveCsv(data_folder+"train/"+csvfilename, tr)
        saveCsv(data_folder+"test/"+csvfilename, ts)
run()