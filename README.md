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
step     0  train 4.1080  val 4.1078
step   100  train 2.0775  val 2.0956
step   200  train 1.6022  val 1.6264
step   300  train 0.8505  val 0.8576
```

## Generate

```
python pakai.py "Selamat pagi pendengar setia"
```

```
python pakai.py --interaktif
```

Flags: `--panjang` (tokens), `--suhu` (default 0.8), `--topk` (default 40), `--ckpt`.

Sample after 300 steps:

```
Selamat pagi pendengar setia, jatan juladi penden yar yang ananak demban dipenger.
z SHai ki sai di -ma jem. ai da sena segalu segari sengar. Kita sejeni k kitalan
betukukuesttn ya, ianga sobempsongensik.
```

Garbled but it has learned the shape of radio language — "Selamat pagi pendengar setia" and "Lagu berikutnya" come out correctly.

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
