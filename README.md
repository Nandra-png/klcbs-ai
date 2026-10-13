# klcbs-ai

A small language model trained from scratch for radio copy. Character-level transformer in plain PyTorch. No pretrained weights, no HuggingFace, no tokenizer library.

The idea: KLCBS is a radio station. The DJs write intros, segues, and playlist notes by hand every day. This is an attempt to train a model small enough to run locally that can draft that copy.

## Files

```
model.py         transformer, ~2.7M params
siapkan_data.py  builds the training corpus
latih.py         training loop
pakai.py         generate text from a checkpoint
```

## Setup

```
pip install -r requirements.txt
```

## Prepare data

```
python siapkan_data.py
```

Generates `data/korpus.txt` and `data/kosakata.json`. The corpus is synthetic — 3800 fake transcripts and playlist notes built from templates. Real use would swap in actual broadcast logs.

Output: ~2.8M characters, 58 unique characters.

## Train

```
python latih.py
```

3000 steps, batch 32, block 128, AdamW at 3e-4. Checkpoints every 250 steps to `klcbs.pt`.

On CPU it takes roughly 50 minutes. Loss curves observed:

```
step     0  train 4.1091  val 4.1082
step   250  train 1.2321  val 1.2448
step   500  train 0.2424  val 0.2465
step   750  train 0.1457  val 0.1470
step  1000  train 0.1219  val 0.1239
step  1500  train 0.1035  val 0.1038
step  2000  train 0.0969  val 0.0988
step  2500  train 0.0968  val 0.0983
step  3000  train 0.0943  val 0.0958
```

## Generate

```
python pakai.py "Selamat pagi pendengar setia"
```

```
python pakai.py --interaktif
```

Flags: `--panjang` (tokens), `--suhu` (default 0.8), `--topk` (default 40), `--ckpt`.

Sample after 3000 steps:

```
Selamat pagi pendengar setia, ini Sore Santai di KLCBS FM bersama Ayu. Gimana
kabarnya hari ini? Semoga semua sehat ya. Sesuai judul segmen kita hari ini,
kita ngobrolin peralatan siaran yang dibutuhkan. Banyak yang salah paham soal
peralatan siaran yang dibutuhkan, padahal sederhana kok. Kita jeda sejenak untuk
pesan dari sponsor. Lagu berikutnya, saya pilih lagu ini karena cocok sama
suasana sekarang.
```

Coherent at this point. It stays on topic across several sentences and keeps the radio register. The 58-character vocab means it cannot spell anything outside the corpus, which shows up as the occasional invented word.

## Architecture

Decoder-only transformer:

- character-level tokenizer, 58 vocab
- 6 layers, 6 heads, 192 embedding dim
- causal self-attention, pre-LayerNorm
- weight tying between token embedding and output head
- GELU feedforward, 4x expansion

Everything is hand-written — multi-head attention, positional embeddings, the sampling loop.

## What's missing

- no BPE tokenizer, characters only
- no early stopping, no LR schedule
- CPU only, no mixed precision
- the corpus is templates, so the model memorizes phrasing rather than composing
- no evaluation beyond loss
