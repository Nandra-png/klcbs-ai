"""Train KLCBS from scratch on the radio corpus.

python latih.py
"""

import json
import os
import time

import torch

from model import KLCBS
from siapkan_data import utama as bikin_korpus

BLOCK = 128
BATCH = 32
STEPS = 3000
LR = 3e-4
EVAL_TIAP = 250
OUT = "klcbs.pt"


def muat_data():
    if not os.path.exists("data/korpus.txt"):
        print("korpus belum ada, bikin dulu")
        bikin_korpus()

    with open("data/korpus.txt", encoding="utf-8") as f:
        teks = f.read()
    with open("data/kosakata.json", encoding="utf-8") as f:
        chars = json.load(f)["chars"]

    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in teks], dtype=torch.long)
    return data, stoi, itos


def ambil_batch(data, block, batch):
    idx = torch.randint(len(data) - block, (batch,))
    x = torch.stack([data[i : i + block] for i in idx])
    y = torch.stack([data[i + 1 : i + block + 1] for i in idx])
    return x, y


def main():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data, stoi, itos = muat_data()
    pisah = int(len(data) * 0.9)
    latih, uji = data[:pisah], data[pisah:]
    print(f"latih {len(latih):,} token, uji {len(uji):,} token")

    model = KLCBS(vocab_size=len(stoi), block_size=BLOCK).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    @torch.no_grad()
    def ukur(split):
        model.eval()
        out = {}
        for nama, d in [("latih", latih), ("uji", uji)]:
            if split and nama != split:
                continue
            losses = torch.zeros(50)
            for i in range(50):
                x, y = ambil_batch(d, BLOCK, BATCH)
                _, loss = model(x.to(device), y.to(device))
                losses[i] = loss.item()
            out[nama] = losses.mean().item()
        model.train()
        return out

    awal = time.time()
    for step in range(STEPS + 1):
        if step % EVAL_TIAP == 0:
            m = ukur(None)
            print(
                f"step {step:5d} | latih {m['latih']:.4f} | uji {m['uji']:.4f} "
                f"| {(time.time() - awal) / 60:.1f} menit"
            )
            torch.save(
                {"model": model.state_dict(), "stoi": stoi, "itos": itos, "block": BLOCK},
                OUT,
            )

        x, y = ambil_batch(latih, BLOCK, BATCH)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"slesai, {time.time() - awal:.0f} detik. bobot disimpen di {OUT}")


if __name__ == "__main__":
    main()
