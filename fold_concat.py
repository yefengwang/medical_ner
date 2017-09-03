import sys
import os
import shutil
from collections import defaultdict


def fold_concat(dirname, new_dirname):
    os.makedirs(new_dirname, exist_ok=True)
    files = [filename for filename in os.listdir(dirname) if filename.startswith("fold_")]
    files.sort()
    for fold_dir in files:
        fold_num = int(fold_dir[5:])
        test_result_filename = os.path.join(dirname, fold_dir, "test_res.txt")
        new_filename = os.path.join("test_res_" + str(fold_num) + ".txt")
        new_filename = os.path.join(new_dirname, new_filename)
        print(test_result_filename, new_filename)
        shutil.copy(test_result_filename, new_filename)
    shutil.copy("hyperparams.py", os.path.join(new_dirname, "hyperparams.py"))


def fold_summary(dirname):
    prf = defaultdict(lambda: {'tp': 0, 'ans': 0, 'act': 0, 'p': 0, 'r': 0, 'f': 0})
    for filename in os.listdir(dirname):
        if not filename.endswith(".txt"):
            continue
        test_result_filename = os.path.join(dirname, filename)
        with open(test_result_filename) as f:
            for line in f:
                row = line.split()
                label, tp, ans, act, p, r, f = row
                p, r, f = float(p), float(r), float(f)
                tp, ans, act = int(tp), int(ans), int(act)
                prf[label]['tp'] += tp
                prf[label]['ans'] += ans
                prf[label]['act'] += act
                prf[label]['p'] += p
                prf[label]['r'] += r
                prf[label]['f'] += f
    for label in prf:
        tp = prf[label]['tp']
        ans = prf[label]['ans']
        act = prf[label]['act']
        p = float(tp) / ans
        r = float(tp) / act
        f = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.0
        print("{}\t{}\t{}\t{}\t{:04.2f}\t{:04.2f}\t{:04.2f}".format(label, tp, ans, act, p*100.0, r*100.0, f*100.0))


if __name__ == "__main__":
    fold_concat(sys.argv[1], sys.argv[2])
    fold_summary(sys.argv[2])
