import os, re, sys

ROOT = "/home/zypca/mathmodel/Essays"
DRY = "--dry" in sys.argv

WM_FULL = ["竞赛零距离", "中国大学生在线"]
UNSAFE = {"在线", "大学生", "距离", "零距", "竞赛", "学生在线"}

def variants(min_len):
    vs = set()
    for w in WM_FULL:
        for i in range(len(w)):
            for j in range(i + min_len, len(w) + 1):
                sub = w[i:j]
                if sub not in UNSAFE and len(sub) >= min_len:
                    vs.add(sub)
    return sorted(vs, key=len, reverse=True)

V3 = variants(3)  # 行尾/独立行/短token 场景

RE_LONE = re.compile(r"^#{0,3}\s*[\u4e00-\u9fff]{0,2}(竞赛零距离|中国大学生在线)(\s*[-—\w.]{0,8})?\s*$")
RE_GOV = re.compile(r"^[\w.]{0,4}gov\.cn$")
RE_ZEROD = re.compile(r"^[\w.-]{0,3}ZeroDistance[\w.-]{0,4}$", re.I)

def clean_line(line):
    if not line.strip():
        return line  # 空行必须原样保留
    s = line.rstrip()
    trailing_ws = line[len(s):]
    if RE_LONE.match(s):
        return None
    # 行尾水印（可叠多次）
    changed = True
    while changed:
        changed = False
        for v in V3:
            if s.endswith(v):
                s = s[: -len(v)].rstrip()
                changed = True
                break
    if not s.strip():
        return None
    # 行中 token 级处理
    out_tokens = []
    for tk in s.split(" "):
        if not tk:
            out_tokens.append(tk)
            continue
        hit = next((v for v in V3 if v in tk), None)
        if hit:
            pre, suf = tk.split(hit, 1)
            if len(pre) <= 3 and len(suf) <= 8:
                continue  # 整 token 是水印+OCR噪声，删除
            if suf == "":
                tk = pre  # 水印黏在代码尾部：in竞赛零距离 -> in
                if not tk:
                    continue
        elif RE_GOV.match(tk) or RE_ZEROD.match(tk):
            continue  # www.moe.gov.cn 碎片 / ZeroDistance 罗马字水印
        out_tokens.append(tk)
    return " ".join(out_tokens) + trailing_ws

def main():
    stats = {"rm": 0, "mod": 0, "files": 0}
    per_file = []
    for dirpath, _, files in os.walk(ROOT):
        for fn in sorted(files):
            if not fn.endswith(".md"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p) as f:
                lines = f.read().split("\n")
            out, rm, mod = [], 0, 0
            for line in lines:
                nl = clean_line(line)
                if nl is None:
                    rm += 1
                else:
                    if nl != line:
                        mod += 1
                    out.append(nl)
            if rm or mod:
                stats["files"] += 1
                stats["rm"] += rm
                stats["mod"] += mod
                per_file.append((os.path.relpath(p, ROOT), rm, mod))
                if not DRY:
                    with open(p, "w") as f:
                        f.write("\n".join(out))
    print(f"{'DRY-RUN' if DRY else 'APPLIED'}: {stats['files']} files, "
          f"{stats['rm']} lines removed, {stats['mod']} lines modified")
    for r, rm, mod in sorted(per_file, key=lambda x: -(x[1] + x[2]))[:15]:
        print(f"  {rm:3d} del {mod:3d} mod  {r}")

if __name__ == "__main__":
    main()
