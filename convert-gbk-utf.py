import sys
import shutil
import os

os.makedirs(sys.argv[2], exist_ok=True)

for filename in os.listdir(sys.argv[1]):
    with open(os.path.join(sys.argv[1], filename), "r", encoding="gbk") as source:
        with open(os.path.join(sys.argv[2], filename), "w", encoding="utf-8") as target:
            shutil.copyfileobj(source, target)