import os
from functools import cmp_to_key


def comp(item1, item2):
    item1 = int(item1.split(".")[0][18:])
    item2 = int(item2.split(".")[0][18:])
    return item1 - item2


files = [x for x in os.listdir(".") if "env-w2-distances-" in x]
files.sort(key=cmp_to_key(comp))
out = []
for idx, fname in enumerate(files):
    print(fname)
    with open(fname, "r") as f:
        lines = f.readlines()
        if idx == 0:
            out.append(lines[0])
        out.append(lines[1] + "\n")

with open("env-w2-distances.csv", "w") as f:
    f.writelines(out)

print("done")
