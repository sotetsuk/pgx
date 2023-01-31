import csv

SAMPLES = []
with open("workspace/contractbridge-ddstable-sample100.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i in reader:
        SAMPLES.append(i)

print(SAMPLES)
