"""Print a notebook's text outputs, for checking prose against results: python tools/dump_outputs.py <nb>"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open(sys.argv[1], encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    print("=" * 10, i, src[:70].replace("\n", " | "))
    for o in c.get("outputs", []):
        if "text" in o:
            t = o["text"]
            print(("".join(t) if isinstance(t, list) else t)[:3000])
        elif "data" in o and "text/plain" in o["data"] and "image/png" not in o["data"]:
            t = o["data"]["text/plain"]
            print(("".join(t) if isinstance(t, list) else t)[:3000])
        elif o.get("output_type") == "error":
            print("ERROR", o["ename"], o["evalue"][:500])
