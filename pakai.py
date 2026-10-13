"""Generate radio copy with a trained checkpoint.

python pakai.py "selamat pagi"
python pakai.py --interaktif
"""

import argparse
import json
import os

import torch

from model import KLCBS


def muat(ckpt):
    if not os.path.exists(ckpt):
        raise SystemExit(f"{ckpt} gak ada, jalanin latih.py dulu")
    simpan = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = KLCBS(vocab_size=len(simpan["stoi"]), block_size=simpan["block"])
    model.load_state_dict(simpan["model"])
    model.eval()
    return model, simpan["stoi"], simpan["itos"], simpan["block"]


def tulis(model, stoi, itos, block, awalan, panjang, suhu, top_k):
    if not awalan:
        awalan = "\n"
    idx = torch.tensor([[stoi.get(c, 0) for c in awalan]], dtype=torch.long)
    out = model.generate(idx, panjang, temperature=suhu, top_k=top_k)
    return "".join(itos[int(i)] for i in out[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("awalan", nargs="?", default="", help="teks pembuka")
    p.add_argument("--panjang", type=int, default=400)
    p.add_argument("--suhu", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--ckpt", default="klcbs.pt")
    p.add_argument("--interaktif", action="store_true")
    a = p.parse_args()

    model, stoi, itos, block = muat(a.ckpt)
    print(f"model siap, {len(stoi)} karakter\n")

    if a.interaktif:
        print("ketik awalan terus enter. kosongin buat keluar.\n")
        while True:
            try:
                s = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not s:
                break
            print(tulis(model, stoi, itos, block, s, a.panjang, a.suhu, a.topk))
            print()
    else:
        print(tulis(model, stoi, itos, block, a.awalan, a.panjang, a.suhu, a.topk))


if __name__ == "__main__":
    main()
