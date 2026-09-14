#!/usr/bin/env python3
"""
Сравнение движков распознавания на одной и той же записи.

Записывает голос (или берёт готовый wav), прогоняет через несколько моделей
и печатает результаты рядом — с временем работы. Одна запись, честное
сравнение: разница в тексте объясняется моделью, а не тем, что вы дважды
сказали по-разному.

    python compare_engines.py --record 30
    python compare_engines.py --wav sample.wav
    python compare_engines.py --wav sample.wav --engines sherpa-v2 gigaam-v3-rnnt

Для моделей v3 нужен onnx-asr:
    pip install "onnx-asr[cpu,hub]"
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

import govorun_pc as G

DEFAULT_ENGINES = ["sherpa-v2", "gigaam-v3-ctc", "gigaam-v3-rnnt"]


def record(seconds: float, device: int | None) -> np.ndarray:
    import sounddevice as sd

    print(f"🎤 Пишу {seconds:.0f} секунд. Говорите после гудка...\n")
    for i in (3, 2, 1):
        print(f"   {i}...", end="", flush=True)
        time.sleep(1)
    print("  ПОШЛА ЗАПИСЬ")

    frames = sd.rec(int(seconds * G.SAMPLE_RATE), samplerate=G.SAMPLE_RATE,
                    channels=1, dtype="float32", device=device)
    sd.wait()
    print("⏹  Готово.\n")
    return frames.flatten().astype(np.float32)


def save_wav(audio: np.ndarray, path: Path) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(G.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit("Поддерживается только 16-битный wav")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        audio = (data / 32768.0).astype(np.float32)
        if w.getframerate() != G.SAMPLE_RATE:
            sys.exit(f"Нужен wav с частотой {G.SAMPLE_RATE} Гц, "
                     f"а здесь {w.getframerate()}")
    return audio


def build(engine: str):
    if engine == "sherpa-v2":
        return G._SherpaV2()
    return G._OnnxV3(engine)


def run(engine_name: str, audio: np.ndarray, chunks: list[np.ndarray]) -> dict:
    print(f"\n{'=' * 70}\n▶  {engine_name}\n{'=' * 70}")
    t0 = time.perf_counter()
    try:
        eng = build(engine_name)
    except SystemExit:
        print("   пропускаю — движок недоступен")
        return {}
    except Exception as e:
        print(f"   пропускаю — {type(e).__name__}: {e}")
        return {}
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = []
    for i, chunk in enumerate(chunks, 1):
        print(f"   фрагмент {i}/{len(chunks)}...", end="\r", flush=True)
        try:
            piece = eng.transcribe(chunk)
        except Exception as e:
            print(f"\n   [!] фрагмент {i}: {type(e).__name__}: {e}")
            continue
        if piece:
            parts.append(piece)
    work_time = time.perf_counter() - t0
    text = " ".join(parts).strip()

    dur = audio.size / G.SAMPLE_RATE
    print(" " * 40, end="\r")
    print(f"   загрузка модели: {load_time:6.1f} с")
    print(f"   распознавание:   {work_time:6.1f} с  "
          f"({work_time / dur:.2f}× от длины записи)")
    print(f"   символов:        {len(text):6d}\n")
    print(text if text else "(пусто)")
    return {"engine": engine_name, "load": load_time,
            "work": work_time, "text": text}


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение движков распознавания")
    ap.add_argument("--record", type=float, metavar="СЕК",
                    help="записать столько секунд с микрофона")
    ap.add_argument("--wav", type=Path, help="взять готовый wav 16 кГц")
    ap.add_argument("--device", type=int, default=None, help="номер микрофона")
    ap.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES,
                    help=f"что сравнивать (по умолчанию: {' '.join(DEFAULT_ENGINES)})")
    ap.add_argument("--keep", type=Path, default=Path("sample.wav"),
                    help="куда сохранить запись")
    args = ap.parse_args()

    if args.wav:
        audio = load_wav(args.wav)
        print(f"📂 {args.wav} — {audio.size / G.SAMPLE_RATE:.1f} с")
    elif args.record:
        audio = record(args.record, args.device)
        save_wav(audio, args.keep)
        print(f"💾 Запись сохранена: {args.keep}")
        print("   Можно переиграть сравнение на ней же: "
              f"--wav {args.keep}\n")
    else:
        ap.error("укажите --record СЕК или --wav ФАЙЛ")

    chunks = G.split_audio(audio)
    print(f"✂️  Нарезано фрагментов: {len(chunks)} "
          f"({', '.join(f'{c.size / G.SAMPLE_RATE:.1f}с' for c in chunks)})")

    results = [r for name in args.engines
               if (r := run(name, audio, chunks))]

    if len(results) < 2:
        print("\nСравнивать не с чем — заработал только один движок.")
        return

    print(f"\n{'=' * 70}\n  ИТОГО\n{'=' * 70}")
    dur = audio.size / G.SAMPLE_RATE
    print(f"{'движок':<22}{'загрузка':>10}{'работа':>10}{'× длины':>10}")
    for r in results:
        print(f"{r['engine']:<22}{r['load']:>9.1f}с{r['work']:>9.1f}с"
              f"{r['work'] / dur:>9.2f}×")

    print("\nТексты различаются — читайте выше и выбирайте сами.")
    print("Скорость решает не всё: разница в одно-два слова на абзац важнее,")
    print("чем лишняя секунда обработки.")


if __name__ == "__main__":
    main()
