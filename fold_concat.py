import sys
import os
import shutil
from collections import defaultdict

def fold_concat(dirname, new_dirname):
    os.makedirs(new_dirname, exist_ok=True)
    prf = defaultdict(lambda: {'tp': 0, 'ans': 0, 'act': 0, 'p': 0, 'r': 0, 'f': 0})
    files = [filename for filename in os.listdir(dirname) if filename.startswith("fold_")]
    files.sort()
    for fold_dir in files:
        fold_num = int(fold_dir[5:])
        test_result_filename = os.path.join(dirname, fold_dir, "test_res.txt")
        new_filename = os.path.join("test_res_" + str(fold_num) + ".txt")
        new_filename = os.path.join(new_dirname, new_filename)
        #print(test_result_filename, os.path.join(new_dirname, new_filename))
        print(test_result_filename, new_filename)
        shutil.copy(test_result_filename, new_filename)

if __name__ == "__main__":
    fold_concat(sys.argv[1], sys.argv[2])