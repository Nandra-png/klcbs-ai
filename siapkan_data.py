"""Generate the training corpus.

Fake radio transcripts, jingle copy, and playlist notes. Everything here is
made up. Real training would swap this for actual broadcast logs.
"""

import json
import os
import random

random.seed(42)

PENYIAR = ["Bram", "Sinta", "Dimas", "Rani", "Yoga", "Ayu", "Fajar", "Nadya"]

PROGRAM = [
    "Pagi Ceria", "Sore Santai", "Malam Melodi", "Kelas Musik",
    "Jalur Nostalgia", "Radio Sehat", "Info Lalu Lintas", "Bedah Lagu",
]

GENRE = ["pop", "dangdut", "jazz", "rock", "keroncong", "reggae", "kpop", "indie"]

KOTA = ["Bandung", "Jakarta", "Surabaya", "Yogyakarta", "Medan", "Semarang", "Makassar", "Denpasar"]

TOPIK = [
    "perkembangan musik indonesia",
    "cara memilih lagu untuk program pagi",
    "sejarah radio di indonesia",
    "perbedaan genre musik",
    "tips jadi penyiar yang enak didengar",
    "cara mengatur playlist harian",
    "peralatan siaran yang dibutuhkan",
    "etika siaran langsung",
    "cara membaca iklan radio",
    "mengelola request pendengar",
]

PEMBUKA = [
    "Selamat pagi pendengar setia, ini {program} di KLCBS FM bersama {penyiar}.",
    "Halo semuanya, sudah jam {jam} dan kalian masih bareng {penyiar} di {program}.",
    "Selamat sore warga {kota}, kembali lagi di {program} bersama saya {penyiar}.",
    "Hai hai hai, {penyiar} di sini masih setia nemenin kalian di {program}.",
]

SAPAAN = [
    "Gimana kabarnya hari ini? Semoga semua sehat ya.",
    "Buat kalian yang masih di jalan, hati-hati di jalan ya.",
    "Sudah pada sarapan belum nih?",
    "Yang masih ngantuk, sini merapat dulu.",
]

SEGMEN = [
    "Kali ini kita bakal bahas soal {topik}.",
    "Sesuai judul segmen kita hari ini, kita ngobrolin {topik}.",
    "Nah jadi tadi ada pendengar yang nanya soal {topik}.",
    "Sebelum lanjut ke lagu berikutnya, kita bahas dulu {topik}.",
]

JAWABAN = [
    "Jadi sebenarnya kalau soal {topik}, yang paling penting itu konsisten.",
    "Kalo menurut saya pribadi sih, {topik} itu balik lagi ke selera masing-masing.",
    "Di dunia radio, {topik} memang sering jadi bahan diskusi.",
    "Banyak yang salah paham soal {topik}, padahal sederhana kok.",
]

LAGU = [
    "lagu ini dari album terbaru mereka",
    "ini requestan dari pendengar di {kota}",
    "lagu ini tahun {tahun} tapi masih enak didengar",
    "saya pilih lagu ini karena cocok sama suasana sekarang",
]

TUTUP = [
    "Segitu dulu {program} kali ini, sampai jumpa besok jam yang sama.",
    "Kita ketemu lagi di {program} besok ya, hati-hati semua.",
    "Terima kasih sudah menemani, ini {penyiar} pamit undur diri.",
]

IKLAN = [
    "Iklan dulu sebentar ya, jangan kemana-mana.",
    "Break dulu, habis ini lanjut lagi.",
    "Kita jeda sejenak untuk pesan dari sponsor.",
]


def pilih(lst):
    return random.choice(lst)


def bikin_transkrip():
    penyiar = pilih(PENYIAR)
    program = pilih(PROGRAM)
    kota = pilih(KOTA)
    jam = random.choice(["6 pagi", "7 pagi", "4 sore", "8 malam", "9 malam"])
    tahun = random.randint(1995, 2024)

    bagian = []

    bagian.append(pilih(PEMBUKA).format(program=program, penyiar=penyiar, jam=jam, kota=kota))
    bagian.append(pilih(SAPAAN))

    for _ in range(random.randint(2, 4)):
        topik = pilih(TOPIK)
        bagian.append(pilih(SEGMEN).format(topik=topik))
        bagian.append(pilih(JAWABAN).format(topik=topik))
        if random.random() < 0.4:
            bagian.append(pilih(IKLAN))
        bagian.append(
            "Lagu berikutnya, {lagu}.".format(lagu=pilih(LAGU).format(kota=kota, tahun=tahun))
        )

    bagian.append(pilih(TUTUP).format(program=program, penyiar=penyiar))

    return " ".join(bagian)


def bikin_catatan():
    return (
        "Playlist {program} {genre} - {kota} - {n} lagu, "
        "pengisi acara {penyiar}, durasi 2 jam.".format(
            program=pilih(PROGRAM), genre=pilih(GENRE), kota=pilih(KOTA),
            n=random.randint(12, 20), penyiar=pilih(PENYIAR),
        )
    )


def utama():
    teks = []
    for _ in range(3000):
        teks.append(bikin_transkrip())
    for _ in range(800):
        teks.append(bikin_catatan())

    random.shuffle(teks)
    isi = "\n".join(teks)

    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "korpus.txt"), "w", encoding="utf-8") as f:
        f.write(isi)

    kosakata = sorted(set(isi))
    with open(os.path.join("data", "kosakata.json"), "w", encoding="utf-8") as f:
        json.dump({"chars": kosakata, "size": len(kosakata)}, f, ensure_ascii=False, indent=2)

    print(f"{len(teks)} dokumen")
    print(f"{len(isi):,} karakter")
    print(f"{len(kosakata)} karakter unik")


if __name__ == "__main__":
    utama()
