import sys
import shutil

with open(sys.argv[1], "r", encoding="gbk") as source:
    with open(sys.argv[2], "w", encoding="utf-8") as target:
        shutil.copyfileobj(source, target)