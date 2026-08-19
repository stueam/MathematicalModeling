import os, re, sys

ROOT = "/home/zypca/mathmodel/Essays"
DRY = "--dry" in sys.argv

RE_HEADER = re.compile(r"^#\s*20\d\d\s*高教社杯全国大学生数学建模竞赛\s*$")
RE_H1 = re.compile(r"^#\s+")
RE_DOCIN = re.compile(r"^#{0,3}\s*(doc\s*豆丁\s*www\.docin\.com|www\.docin\.com)\s*$", re.I)
RE_REVIEW_LINE = re.compile(r"^(赛区评阅编号|赛区评阅记录|全国统一编号|全国评阅编号)[（(]?\s*(由.{0,20}编号|可供赛区评阅时使用)?\s*[）)]?[:：]?\s*$")
RE_PAGENO = re.compile(r"^\s*[-—–]?\s*\d{1,3}\s*[-—–]?\s*$")

stats = {"header": 0, "docin": 0, "review": 0, "pageno": 0, "files": 0}

def clean(path):
    with open(path) as f:
        lines = f.read().split("\n")
    out, i, changed = [], 0, False
    in_code = False
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line); i += 1; continue
        if RE_DOCIN.match(line.strip()):
            stats["docin"] += 1; changed = True; i += 1; continue
        if RE_HEADER.match(line.strip()):
            j = i + 1
            while j < n and j - i <= 60:
                s = lines[j].strip()
                if RE_H1.match(s) and not RE_HEADER.match(s):
                    break
                j += 1
            else:
                j = i + 1
            if j < n and RE_H1.match(lines[j].strip()):
                stats["header"] += j - i; changed = True
                i = j; continue
        if not in_code and RE_REVIEW_LINE.match(line.strip()):
            stats["review"] += 1; changed = True; i += 1; continue
        if not in_code and RE_PAGENO.match(line):
            stats["pageno"] += 1; changed = True; i += 1; continue
        out.append(line); i += 1
    while out and not out[-1].strip():
        out.pop()
    text = "\n".join(out) + "\n"
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    if changed:
        stats["files"] += 1
        if not DRY:
            with open(path, "w") as f:
                f.write(text)
    return changed

total = 0
for dirpath, _, files in os.walk(ROOT):
    for fn in files:
        if fn.endswith(".md"):
            total += 1
            clean(os.path.join(dirpath, fn))
print(f"{'DRY-RUN' if DRY else 'APPLIED'}: {total} files, {stats['files']} modified")
print(f"  removed: header blocks {stats['header']} lines, docin {stats['docin']}, review {stats['review']}, pageno {stats['pageno']}")
