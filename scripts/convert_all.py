import io, os, sys, time, zipfile, traceback
from concurrent.futures import ThreadPoolExecutor
import requests

API = "http://127.0.0.1:8260/file_parse"
ROOT = "/home/zypca/mathmodel/Essays"
WORKERS = 1

def convert(pdf):
    stem = pdf[:-4]
    out_md = stem + ".md"
    if os.path.exists(out_md):
        return (pdf, "skip", 0)
    for attempt in (1, 2):
        try:
            with open(pdf, "rb") as f:
                r = requests.post(
                    API,
                    files=[("files", (os.path.basename(pdf), f, "application/pdf"))],
                    data={"backend": "pipeline", "return_md": "true",
                          "return_images": "true", "response_format_zip": "true"},
                    timeout=3600)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            md_name = next(x for x in z.namelist() if x.endswith(".md"))
            md = z.read(md_name).decode("utf-8")
            img_dir = os.path.join(os.path.dirname(pdf), "images", os.path.basename(stem))
            os.makedirs(img_dir, exist_ok=True)
            n_img = 0
            for name in z.namelist():
                base = os.path.basename(name)
                if not name.endswith(".md") and base:
                    with open(os.path.join(img_dir, base), "wb") as g:
                        g.write(z.read(name))
                    md = md.replace(f"images/{base}", f"images/{os.path.basename(stem)}/{base}")
                    n_img += 1
            with open(out_md, "w") as f:
                f.write(md)
            return (pdf, "ok", n_img)
        except Exception as e:
            if attempt == 2:
                return (pdf, f"error: {type(e).__name__} {e}", 0)
            time.sleep(10)

if __name__ == "__main__":
    pdfs = []
    for dirpath, _, files in os.walk(ROOT):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, fn))
    pdfs.sort()
    print(f"{len(pdfs)} pdfs, workers={WORKERS}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, (pdf, status, n) in enumerate(ex.map(convert, pdfs), 1):
            print(f"[{i}/{len(pdfs)}] {status} ({n} img) {os.path.relpath(pdf, ROOT)}", flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.0f} min", flush=True)
