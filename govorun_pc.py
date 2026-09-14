#!/usr/bin/env python3
"""
Govorun PC — голосовой ввод на русском для Windows.
Офлайн, на основе GigaAM v2 (Сбер) + sherpa-onnx.

Использование:
    python govorun_pc.py               # запуск с иконкой в трее
    python govorun_pc.py --device 2    # выбрать микрофон
    python govorun_pc.py --list-devices

Требования:
    pip install -r requirements.txt
    python download_models.py          # один раз, ~330 МБ
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Перенаправляем кэши PyTorch/HuggingFace в домашнюю папку
# (до любых импортов torch/transformers)
_CACHE = Path.home() / ".cache" / "govorun"
os.environ.setdefault("HF_HOME",            str(_CACHE / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE",  str(_CACHE / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME",          str(_CACHE / "torch"))

import numpy as np
import sounddevice as sd
import sherpa_onnx
import keyboard
import pyperclip
import pystray
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Конфигурация (изменяется через настройки)
# ============================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
MODELS_DIR  = SCRIPT_DIR / "models"
MODEL_PATH  = MODELS_DIR / "model.onnx"
TOKENS_PATH = MODELS_DIR / "tokens.txt"

SAMPLE_RATE     = 16_000
FEATURE_DIM     = 80
MIN_DURATION_SEC = 0.3

HOTKEY: str       = "alt+x"
INPUT_DEVICE: int | None = None

# ---- Нарезка длинного аудио -------------------------------------------------
# GigaAM обучена на сегментах примерно до 25 секунд. Всё, что длиннее, уходит
# за пределы её контекста, и качество разваливается тем сильнее, чем дальше
# от начала записи. Поэтому режем запись по паузам на куски.
MAX_CHUNK_SEC   = 18.0   # целевой максимум одного куска
MIN_CHUNK_SEC   = 1.0    # короче — приклеиваем к соседнему
SILENCE_MIN_SEC = 0.35   # пауза такой длины считается границей фразы
FRAME_MS        = 20     # окно анализа громкости

# Задержка между открытием микрофона и появлением значка записи.
# Поток звука поднимается не мгновенно; без паузы срезает первые слова.
WARMUP_SEC      = 0.25

# Куда падает распознанный текст, если вставка не сработала
FALLBACK_DIR    = Path.home() / ".govorun"

# Настройки и словарь замен. Лежат рядом со скриптом, правятся руками.
CONFIG_PATH       = SCRIPT_DIR / "config.json"
REPLACEMENTS_PATH = SCRIPT_DIR / "replacements.txt"

_cpu = os.cpu_count() or 4
NUM_THREADS = max(2, min(8, _cpu - 2))

USE_PUNCT: bool = True   # пишется в config.json


# ============================================================
# Настройки: живут между запусками
# ============================================================
def load_config() -> None:
    """Читает config.json. Отсутствие файла — не ошибка, просто дефолты."""
    global HOTKEY, INPUT_DEVICE, USE_PUNCT
    if not CONFIG_PATH.exists():
        return
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  config.json повреждён, беру значения по умолчанию: {e}")
        return

    HOTKEY = str(data.get("hotkey", HOTKEY)).strip().lower() or HOTKEY
    USE_PUNCT = bool(data.get("punctuation", USE_PUNCT))

    # Микрофон храним по имени: индексы съезжают при перетыкании устройств
    name = data.get("input_device_name")
    if name:
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and d["name"] == name:
                    INPUT_DEVICE = i
                    break
            else:
                print(f"⚠️  Микрофон «{name}» не найден, беру системный по умолчанию")
        except Exception:
            pass


def save_config() -> None:
    # Имя микрофона выясняем отдельно: если устройство выдернули,
    # это не повод терять остальные настройки
    name = None
    if INPUT_DEVICE is not None:
        try:
            name = sd.query_devices(INPUT_DEVICE)["name"]
        except Exception:
            pass

    try:
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "hotkey": HOTKEY,
                    "input_device_name": name,
                    "punctuation": USE_PUNCT,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[!] Не удалось сохранить настройки: {e}")


# ============================================================
# Словарь замен: «как слышится» → «как пишется»
# ============================================================
_replacements: list[tuple[re.Pattern, str]] = []
_replacements_mtime: float = -1.0

# Падежные и числовые окончания. Список закрытый и упорядочен от длинных
# к коротким, иначе «а» съест начало «ами».
_RU_ENDINGS = (
    r"(?:ами|ыми|ах|ях|ам|ям|ов|ев|ом|ем|ой|ей|ою|ые|ых|ым|ие|ий|ия|ию|"
    r"а|у|е|ы|и|о|ю|я)?"
)
_CYRILLIC_ONLY  = re.compile(r"[а-яё]+", re.IGNORECASE)
_TRAILING_VOWEL = re.compile(r"[аеёиоуыэюя]$", re.IGNORECASE)

REPLACEMENTS_TEMPLATE = """\
# Словарь замен для Govorun.
# GigaAM обучена на русском и латиницу не выдаёт в принципе: «Warhammer»
# у неё всегда «вархамер». Здесь задаётся, как такие слова писать.
#
# Формат:  что слышим = как писать
# Строки с # и пустые игнорируются. Регистр слева не важен.
# Файл перечитывается на лету — перезапуск не нужен.
# Сначала применяются самые длинные фразы, так что «самсунг гэлакси»
# сработает раньше, чем отдельный «самсунг».

вархамер = Warhammer
самсунг гэлакси = Samsung Galaxy
майкрософт = Microsoft
интел = Intel
кейкрон = Keychron
ноушн = Notion
ноушен = Notion
клод = Claude
гитхаб = GitHub
виндовс = Windows
питон = Python
"""


def load_replacements(verbose: bool = True) -> None:
    """
    Читает replacements.txt. Файла нет — создаём с примерами.
    Перечитывает только при изменении файла, так что дёргать можно часто.
    """
    global _replacements, _replacements_mtime

    if not REPLACEMENTS_PATH.exists():
        try:
            REPLACEMENTS_PATH.write_text(REPLACEMENTS_TEMPLATE, encoding="utf-8")
            if verbose:
                print(f"📖 Создан словарь замен: {REPLACEMENTS_PATH.name}")
        except Exception as e:
            print(f"[!] Не удалось создать словарь замен: {e}")
            return

    try:
        mtime = REPLACEMENTS_PATH.stat().st_mtime
    except OSError:
        return
    if mtime == _replacements_mtime:
        return
    _replacements_mtime = mtime

    pairs: list[tuple[str, str]] = []
    try:
        for line in REPLACEMENTS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            src, dst = line.split("=", 1)
            src, dst = src.strip(), dst.strip()
            if src:
                pairs.append((src, dst))
    except Exception as e:
        print(f"[!] Не удалось прочитать словарь замен: {e}")
        return

    # Длинные фразы первыми, иначе «самсунг» съест «самсунг гэлакси»
    pairs.sort(key=lambda p: len(p[0]), reverse=True)

    compiled = []
    for src, dst in pairs:
        words = src.split()
        last  = words[-1]

        # Русский склоняется: «в ноушене», «на прусаслайсере», «из гитхаба».
        # Разрешаем падежное окончание, но только из закрытого списка —
        # иначе «стим» начнёт ловить «стимул».
        if _CYRILLIC_ONLY.fullmatch(last):
            # Если слово записано уже с окончанием («синджента», «джира»),
            # в косвенных падежах оно заменяется, а не дописывается.
            # Поэтому конечную гласную отрезаем. Короткие корни не трогаем:
            # из «кура» вышло бы «кур», и правило поймало бы «куры».
            stem = _TRAILING_VOWEL.sub("", last)
            if len(stem) < 4:
                stem = last
            tail = _RU_ENDINGS
        else:
            stem = last
            tail = ""

        parts = [re.escape(w) for w in words[:-1]] + [re.escape(stem)]
        # \b по краям, чтобы «интел» не портил «интеллект»
        pattern = r"\b" + r"\s+".join(parts) + tail + r"\b"

        try:
            compiled.append((re.compile(pattern, re.IGNORECASE | re.UNICODE), dst))
        except re.error as e:
            print(f"[!] Строка словаря пропущена ({src}): {e}")

    _replacements = compiled
    if verbose:
        print(f"📖 Словарь замен: {len(_replacements)} правил\n")


def apply_replacements(text: str) -> str:
    if not text:
        return text
    load_replacements(verbose=False)   # подхватываем правки на лету
    for pattern, dst in _replacements:
        text = pattern.sub(dst, text)
    return text


# ============================================================
# Иконки трея (рисуются через Pillow)
# ============================================================
def _make_icon(recording: bool) -> Image.Image:
    size = 64
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Фон: серый в покое, красный при записи
    bg = (210, 40, 40, 255) if recording else (60, 60, 60, 255)
    draw.ellipse([2, 2, 62, 62], fill=bg)

    # Тело микрофона
    draw.rounded_rectangle([24, 10, 40, 36], radius=8, fill="white")

    # Дуга-подставка
    draw.arc([18, 28, 46, 50], start=180, end=0, fill="white", width=3)

    # Ножка
    draw.line([32, 50, 32, 57], fill="white", width=3)

    # Основание
    draw.line([24, 57, 40, 57], fill="white", width=3)

    # Мигающая точка при записи
    if recording:
        draw.ellipse([46, 4, 60, 18], fill=(255, 80, 80, 255))

    return img


ICON_IDLE      = _make_icon(False)
ICON_RECORDING = _make_icon(True)


# ============================================================
# Запись звука
# ============================================================
class Recorder:
    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.frames: list[np.ndarray] = []
        self.recording = False
        self.stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self.level = 0.0          # текущая громкость 0..1 для индикатора

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[!] sounddevice: {status}")
        with self._lock:
            if self.recording:
                self.frames.append(indata.copy())
        # Считаем уровень всегда, даже вне записи — дешёво и нужно для индикатора
        try:
            rms = float(np.sqrt(np.mean(indata.astype(np.float64) ** 2)))
        except Exception:
            rms = 0.0
        # Логарифмическая шкала: линейная почти всегда выглядит нулём
        self.level = min(1.0, max(0.0, (np.log10(rms + 1e-6) + 3.5) / 3.0))

    def start(self) -> None:
        with self._lock:
            self.frames = []
            self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=INPUT_DEVICE,
            latency="low",
            callback=self._callback,
        )
        self.stream.start()
        # Даём потоку подняться до того, как пользователь увидит значок записи
        time.sleep(WARMUP_SEC)

    def _close(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.level = 0.0

    def cancel(self) -> None:
        """Бросить запись, ничего не распознавая."""
        with self._lock:
            self.recording = False
            self.frames = []
        self._close()

    def stop(self) -> np.ndarray:
        with self._lock:
            self.recording = False
        self._close()
        with self._lock:
            if not self.frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.frames).flatten().astype(np.float32)


# ============================================================
# Распознавание
# ============================================================
def load_recognizer() -> sherpa_onnx.OfflineRecognizer:
    if not MODEL_PATH.exists() or not TOKENS_PATH.exists():
        print(f"[!] Модель не найдена в {MODELS_DIR}", file=sys.stderr)
        print("    Запустите:  python download_models.py", file=sys.stderr)
        sys.exit(1)
    print(f"⏳ Загружаю GigaAM ({NUM_THREADS} потоков)...")
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(MODEL_PATH),
        tokens=str(TOKENS_PATH),
        num_threads=NUM_THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method="greedy_search",
    )
    print("✅ Модель готова.\n")
    return rec


def _recognize_one(rec: sherpa_onnx.OfflineRecognizer, audio: np.ndarray) -> str:
    """Распознать один кусок, который заведомо помещается в контекст модели."""
    if audio.size / SAMPLE_RATE < MIN_DURATION_SEC:
        return ""
    stream = rec.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    rec.decode_stream(stream)
    return stream.result.text.strip()


def split_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """
    Режет запись на куски не длиннее MAX_CHUNK_SEC, стараясь ставить границы
    в паузах между фразами.

    Порог тишины подбирается под конкретную запись: берём тихие 15% кадров
    как оценку шума помещения и поднимаемся над ней. Так работает и в тишине,
    и при гудящем вентиляторе.
    """
    total_sec = audio.size / sr
    if total_sec <= MAX_CHUNK_SEC:
        return [audio]

    frame = int(sr * FRAME_MS / 1000)
    n_frames = audio.size // frame
    if n_frames == 0:
        return [audio]

    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

    # Нижний перцентиль — оценка фона, верхний — уровень речи.
    # Берём 5-й, а не 15-й: пауз в записи может быть всего несколько процентов,
    # и тогда более высокий перцентиль попадёт уже в речь.
    floor = float(np.quantile(rms, 0.05))
    top   = float(np.quantile(rms, 0.90))

    if top <= floor * 2:
        # Запись ровная по громкости — надёжных пауз не найти.
        # Режем просто по длине, без ложных границ.
        is_silent = np.zeros(n_frames, dtype=bool)
    else:
        threshold = floor + (top - floor) * 0.12
        is_silent = rms < threshold
    min_sil_frames = max(1, int(SILENCE_MIN_SEC * 1000 / FRAME_MS))

    # Середины достаточно длинных пауз — кандидаты в точки реза
    cuts: list[int] = []
    run_start = None
    for i, sil in enumerate(is_silent):
        if sil and run_start is None:
            run_start = i
        elif not sil and run_start is not None:
            if i - run_start >= min_sil_frames:
                cuts.append((run_start + i) // 2 * frame)
            run_start = None
    if run_start is not None and n_frames - run_start >= min_sil_frames:
        cuts.append((run_start + n_frames) // 2 * frame)

    max_len = int(MAX_CHUNK_SEC * sr)
    min_len = int(MIN_CHUNK_SEC * sr)

    chunks: list[np.ndarray] = []
    start = 0
    while start < audio.size:
        limit = start + max_len
        if limit >= audio.size:
            chunks.append(audio[start:])
            break
        # Самая поздняя пауза, которая ещё помещается в лимит
        candidates = [c for c in cuts if start + min_len < c <= limit]
        end = candidates[-1] if candidates else limit
        chunks.append(audio[start:end])
        start = end

    # Хвостовой огрызок приклеиваем к предыдущему куску, чтобы не гонять
    # модель ради полутора секунд и не рвать фразу посередине
    merged: list[np.ndarray] = []
    for c in chunks:
        if merged and c.size < min_len and merged[-1].size + c.size <= max_len * 1.3:
            merged[-1] = np.concatenate([merged[-1], c])
        else:
            merged.append(c)

    return [c for c in merged if c.size >= int(MIN_DURATION_SEC * sr)]


def recognize(
    rec: sherpa_onnx.OfflineRecognizer,
    audio: np.ndarray,
    on_progress=None,
) -> str:
    """Распознать запись целиком, при необходимости разбив её на куски."""
    if audio.size / SAMPLE_RATE < MIN_DURATION_SEC:
        return ""

    chunks = split_audio(audio)
    if len(chunks) > 1:
        print(f"✂️  Запись нарезана на {len(chunks)} фрагм. по паузам")

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        if on_progress:
            on_progress(i, len(chunks))
        try:
            piece = _recognize_one(rec, chunk)
        except Exception as e:
            print(f"[!] Фрагмент {i}/{len(chunks)} не распознан: {e}")
            continue
        if piece:
            parts.append(piece)
            if len(chunks) > 1:
                print(f"   [{i}/{len(chunks)}] {piece}")

    return " ".join(parts).strip()


# ============================================================
# Пунктуация (ONNX, без PyTorch)
# ============================================================
_punct_model = None

def load_punct_model() -> bool:
    global _punct_model
    try:
        from punctuators.models import PunctCapSegModelONNX
        print("⏳ Загружаю модель пунктуации (ONNX)...")
        _punct_model = PunctCapSegModelONNX.from_pretrained(
            "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
        )
        print("✅ Пунктуация готова.\n")
        return True
    except ImportError:
        print("⚠️  punctuators не установлен — пунктуация отключена.\n")
        return False
    except Exception as e:
        print(f"⚠️  Не удалось загрузить модель пунктуации: {e}\n")
        return False


def restore_punctuation(text: str) -> str:
    if not text:
        return text
    if _punct_model is not None:
        try:
            results = _punct_model.infer([text])
            # Модель возвращает список предложений. Склеивать их надо пробелом,
            # иначе получается «чудеса.Там леший бродит».
            if results:
                return " ".join(s.strip() for s in results[0] if s.strip())
            return text
        except Exception as e:
            print(f"[!] Ошибка пунктуации: {e}")
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


# ============================================================
# Вставка текста
# ============================================================
def save_fallback(text: str) -> Path | None:
    """
    Складывает распознанный текст на диск. Вызывается всегда, до попытки
    вставки: если вставка не сработает или программа упадёт, текст не пропадёт.
    """
    try:
        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log = FALLBACK_DIR / "history.txt"
        with log.open("a", encoding="utf-8") as f:
            f.write(f"--- {stamp} ---\n{text}\n\n")
        (FALLBACK_DIR / "last.txt").write_text(text, encoding="utf-8")
        return log
    except Exception as e:
        print(f"[!] Не удалось сохранить резервную копию: {e}")
        return None


def paste_text(text: str) -> bool:
    """
    Кладёт текст в буфер и вставляет в активное окно.
    Возвращает False, если вставка не удалась — текст при этом остаётся
    в буфере, его можно вставить руками через Ctrl+V.
    """
    try:
        pyperclip.copy(text)
    except Exception as e:
        print(f"[!] Буфер обмена недоступен: {e}")
        return False

    time.sleep(0.05)
    try:
        keyboard.send("ctrl+v")
        return True
    except Exception as e:
        print(f"[!] Не удалось вставить: {e}")
        return False


# ============================================================
# Контроллер: Alt+X → старт/стоп
# ============================================================
class Controller:
    def __init__(self, rec: sherpa_onnx.OfflineRecognizer):
        self.rec      = rec
        self.recorder = Recorder()
        self._busy      = False
        self._busy_lock = threading.Lock()
        self.on_state_change: callable = lambda recording: None  # колбэк для трея
        # (номер фрагмента, всего). (0, 0) — обработка закончена
        self.on_progress: callable = lambda done, total: None
        self._esc_hook = None

    def toggle(self, _event=None) -> None:
        with self._busy_lock:
            if self._busy:
                return
        if self.recorder.recording:
            self._unhook_esc()
            with self._busy_lock:
                self._busy = True
            threading.Thread(target=self._process, daemon=True).start()
        else:
            print(f"🎤 Запись... ({HOTKEY.upper()} — стоп, ESC — отмена)")
            self.recorder.start()
            self._hook_esc()
            self.on_state_change(True)

    # --- отмена по Esc --------------------------------------------------
    def _hook_esc(self) -> None:
        """
        Esc перехватывается только на время записи и без suppress —
        в остальное время клавиша работает как обычно.
        """
        try:
            self._esc_hook = keyboard.on_press_key("esc", self._on_esc, suppress=False)
        except Exception as e:
            print(f"[!] Не удалось повесить Esc: {e}")
            self._esc_hook = None

    def _unhook_esc(self) -> None:
        if self._esc_hook is not None:
            try:
                keyboard.unhook(self._esc_hook)
            except Exception:
                pass
            self._esc_hook = None

    def _on_esc(self, _event=None) -> None:
        if not self.recorder.recording:
            return
        self._unhook_esc()
        self.recorder.cancel()
        self.on_state_change(False)
        self.on_progress(0, 0)
        print("✖  Запись отменена\n")

    def _process(self) -> None:
        self.on_state_change(False)
        audio = self.recorder.stop()
        dur   = audio.size / SAMPLE_RATE
        rms   = float(np.sqrt(np.mean(audio ** 2))) if audio.size > 0 else 0.0
        print(f"⏹  Записано {dur:.1f}с, уровень: {rms:.4f}, распознаю...")
        started = time.monotonic()

        def progress(i: int, total: int) -> None:
            if total > 1:
                self.on_progress(i, total)

        try:
            text = recognize(self.rec, audio, on_progress=progress)
        except Exception as e:
            print(f"[!] Ошибка распознавания: {e}")
            text = ""

        self.on_progress(0, 0)

        if text:
            text = restore_punctuation(text)
            text = apply_replacements(text)
            # Сохраняем ДО вставки — тогда даже падение на вставке не съест текст
            save_fallback(text)
            print(f"📝 {text}")
            if paste_text(text):
                print(f"✅ Вставлено за {time.monotonic() - started:.1f}с\n")
            else:
                print("⚠️  Вставить не получилось. Текст в буфере — нажмите Ctrl+V.")
                print(f"   Копия: {FALLBACK_DIR / 'last.txt'}\n")
        else:
            print("(тишина или слишком короткая запись)\n")
        with self._busy_lock:
            self._busy = False


# ============================================================
# Окно настроек (tkinter)
# ============================================================
class SettingsWindow:
    def __init__(self, root: tk.Tk, ctrl: Controller, tray: "TrayApp"):
        self.ctrl = ctrl
        self.tray = tray
        self.win  = tk.Toplevel(root)
        self.win.title("Govorun — настройки")
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        self._build()
        # Центрируем окно
        self.win.update_idletasks()
        w, h = self.win.winfo_width(), self.win.winfo_height()
        x = (self.win.winfo_screenwidth()  - w) // 2
        y = (self.win.winfo_screenheight() - h) // 2
        self.win.geometry(f"{w}x{h}+{x}+{y}")

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}
        f   = tk.Frame(self.win, padx=16, pady=12)
        f.pack(fill="both", expand=True)

        # --- Горячая клавиша ---
        tk.Label(f, text="Горячая клавиша:", anchor="w").grid(
            row=0, column=0, sticky="w", **pad)
        self._hotkey_var = tk.StringVar(value=HOTKEY)
        tk.Entry(f, textvariable=self._hotkey_var, width=20).grid(
            row=0, column=1, sticky="ew", **pad)

        # --- Микрофон ---
        tk.Label(f, text="Микрофон:", anchor="w").grid(
            row=1, column=0, sticky="w", **pad)
        devices     = sd.query_devices()
        self._dev_names = [
            d["name"] for d in devices if d["max_input_channels"] > 0
        ]
        self._dev_ids = [
            i for i, d in enumerate(devices) if d["max_input_channels"] > 0
        ]
        current_name = (
            devices[INPUT_DEVICE]["name"]
            if INPUT_DEVICE is not None
            else devices[sd.default.device[0]]["name"]
        )
        self._device_var = tk.StringVar(value=current_name)
        ttk.Combobox(
            f,
            textvariable=self._device_var,
            values=self._dev_names,
            state="readonly",
            width=30,
        ).grid(row=1, column=1, sticky="ew", **pad)

        # --- Пунктуация ---
        self._punct_var = tk.BooleanVar(value=_punct_model is not None)
        tk.Checkbutton(
            f, text="Восстанавливать пунктуацию", variable=self._punct_var
        ).grid(row=2, columnspan=2, sticky="w", **pad)

        # --- Словарь замен ---
        tk.Button(
            f, text="Открыть словарь замен…", width=24,
            command=self._open_replacements,
        ).grid(row=3, columnspan=2, sticky="w", **pad)

        # --- Кнопки ---
        btn_frame = tk.Frame(f)
        btn_frame.grid(row=4, columnspan=2, pady=(8, 0))
        tk.Button(btn_frame, text="Применить", width=12,
                  command=self._apply).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Отмена",    width=12,
                  command=self.win.destroy).pack(side="left", padx=4)

    def _apply(self) -> None:
        global HOTKEY, INPUT_DEVICE, _punct_model

        # Горячая клавиша
        new_hotkey = self._hotkey_var.get().strip().lower()
        if new_hotkey and new_hotkey != HOTKEY:
            try:
                keyboard.remove_hotkey(HOTKEY)
                keyboard.add_hotkey(new_hotkey, self.ctrl.toggle, suppress=True)
                HOTKEY = new_hotkey
                print(f"⌨️  Хоткей изменён на [{HOTKEY.upper()}]")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось установить хоткей:\n{e}")
                return

        # Микрофон
        chosen_name = self._device_var.get()
        if chosen_name in self._dev_names:
            idx = self._dev_ids[self._dev_names.index(chosen_name)]
            INPUT_DEVICE = idx
            print(f"🎙  Микрофон: [{idx}] {chosen_name}")

        # Пунктуация
        global USE_PUNCT
        want_punct = self._punct_var.get()
        USE_PUNCT = want_punct
        if want_punct and _punct_model is None:
            load_punct_model()
        elif not want_punct:
            _punct_model = None

        save_config()
        print(f"💾 Настройки сохранены в {CONFIG_PATH.name}")

        self.tray.update_tooltip()
        self.win.destroy()

    def _open_replacements(self) -> None:
        """Открывает словарь замен в системном редакторе."""
        load_replacements(verbose=False)   # создаст файл, если его нет
        try:
            os.startfile(str(REPLACEMENTS_PATH))   # только Windows
        except AttributeError:
            import subprocess
            subprocess.Popen(["xdg-open", str(REPLACEMENTS_PATH)])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")


# ============================================================
# Оверлей записи — маленький бейдж поверх всех окон
# ============================================================
class RecordingOverlay:
    W, H     = 168, 36
    BG       = "#1e1e1e"
    DOT_ON   = "#ff3b3b"
    DOT_OFF  = "#7a1010"
    FG       = "#ffffff"
    BAR_BG   = "#3a3a3a"
    BAR_FG   = "#4ade80"   # зелёный: микрофон слышит
    BAR_HOT  = "#facc15"   # жёлтый: близко к перегрузу
    BAR_X0, BAR_X1 = 96, 156
    BAR_H    = 8

    def __init__(self, root: tk.Tk):
        self._root   = root
        self._win: tk.Toplevel | None = None
        self._canvas: tk.Canvas | None = None
        self._dot_id: int | None = None
        self._text_id: int | None = None
        self._bar_id: int | None = None
        self._blink_on = True
        self._blink_job = None
        self._level_job = None
        # Откуда брать громкость микрофона; ставится снаружи
        self.level_provider: callable = lambda: 0.0

    def set_label(self, label: str) -> None:
        """Меняет надпись на бейдже: REC во время записи, 2/5 при обработке."""
        if self._canvas and self._text_id:
            self._canvas.itemconfig(self._text_id, text=label)

    def show(self, label: str = "● REC") -> None:
        if self._win and self._win.winfo_exists():
            self.set_label(label)
            return

        win = tk.Toplevel(self._root)
        win.overrideredirect(True)          # без рамки и заголовка
        win.attributes("-topmost", True)    # поверх всех окон
        win.attributes("-alpha", 0.88)
        win.resizable(False, False)

        # Позиция: правый нижний угол над панелью задач
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x  = sw - self.W - 16
        y  = sh - self.H - 52
        win.geometry(f"{self.W}x{self.H}+{x}+{y}")

        c = tk.Canvas(win, width=self.W, height=self.H,
                      bg=self.BG, highlightthickness=0)
        c.pack()

        # Скруглённый фон
        r = 10
        c.create_arc(0,        0,        2*r, 2*r, start=90,  extent=90,  fill=self.BG, outline=self.BG)
        c.create_arc(self.W-2*r, 0,      self.W, 2*r, start=0, extent=90, fill=self.BG, outline=self.BG)
        c.create_rectangle(r, 0, self.W-r, self.H, fill=self.BG, outline=self.BG)
        c.create_rectangle(0, r, self.W, self.H-r, fill=self.BG, outline=self.BG)

        # Мигающая точка
        dot_x, dot_y = 16, self.H // 2
        self._dot_id = c.create_oval(
            dot_x - 6, dot_y - 6, dot_x + 6, dot_y + 6,
            fill=self.DOT_ON, outline=""
        )

        # Текст
        self._text_id = c.create_text(dot_x + 14, dot_y, anchor="w",
                                      text=label, fill=self.FG,
                                      font=("Segoe UI", 11, "bold"))

        # Индикатор уровня микрофона: видно, что он реально слышит,
        # а не что запись идёт в тишину
        bar_y = self.H // 2
        c.create_rectangle(self.BAR_X0, bar_y - self.BAR_H // 2,
                           self.BAR_X1, bar_y + self.BAR_H // 2,
                           fill=self.BAR_BG, outline="")
        self._bar_id = c.create_rectangle(self.BAR_X0, bar_y - self.BAR_H // 2,
                                          self.BAR_X0, bar_y + self.BAR_H // 2,
                                          fill=self.BAR_FG, outline="")

        self._win    = win
        self._canvas = c
        self._blink_on = True
        self._blink()
        self._level()

    def hide(self) -> None:
        for job in ("_blink_job", "_level_job"):
            j = getattr(self, job)
            if j:
                self._root.after_cancel(j)
                setattr(self, job, None)
        if self._win and self._win.winfo_exists():
            self._win.destroy()
        self._win     = None
        self._canvas  = None
        self._dot_id  = None
        self._text_id = None

    def _blink(self) -> None:
        if self._canvas and self._dot_id:
            color = self.DOT_ON if self._blink_on else self.DOT_OFF
            self._canvas.itemconfig(self._dot_id, fill=color)
            self._blink_on = not self._blink_on
            self._blink_job = self._root.after(500, self._blink)

    def _level(self) -> None:
        """Обновляет полоску громкости. 10 раз в секунду — глазу достаточно."""
        if self._canvas and self._bar_id:
            try:
                lvl = float(self.level_provider())
            except Exception:
                lvl = 0.0
            lvl = min(1.0, max(0.0, lvl))
            y = self.H // 2
            x1 = self.BAR_X0 + (self.BAR_X1 - self.BAR_X0) * lvl
            self._canvas.coords(self._bar_id,
                                self.BAR_X0, y - self.BAR_H // 2,
                                x1, y + self.BAR_H // 2)
            self._canvas.itemconfig(
                self._bar_id, fill=self.BAR_HOT if lvl > 0.9 else self.BAR_FG)
            self._level_job = self._root.after(100, self._level)


# ============================================================
# Трей
# ============================================================
class TrayApp:
    def __init__(self, ctrl: Controller, root: tk.Tk):
        self.ctrl    = ctrl
        self.root    = root
        self.overlay = RecordingOverlay(root)
        self._icon: pystray.Icon | None = None
        ctrl.on_state_change = self._on_state_change
        ctrl.on_progress     = self._on_progress
        self.overlay.level_provider = lambda: ctrl.recorder.level

    def _on_state_change(self, recording: bool) -> None:
        if self._icon is not None:
            self._icon.icon  = ICON_RECORDING if recording else ICON_IDLE
            self._icon.title = "🎤 Запись идёт..." if recording else f"Govorun  [{HOTKEY.upper()}]"
        # Оверлей — в главном потоке tkinter
        if recording:
            self.root.after(0, self.overlay.show)
        else:
            # Бейдж не прячем сразу: он превращается в индикатор обработки,
            # чтобы не стоять и не гадать, работает оно или зависло
            self.root.after(0, lambda: self.overlay.show("⏳ ..."))

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.root.after(0, self.overlay.hide)
        else:
            self.root.after(0, lambda: self.overlay.set_label(f"⏳ {done}/{total}"))

    def update_tooltip(self) -> None:
        if self._icon:
            self._icon.title = f"Govorun  [{HOTKEY.upper()}]"

    def _open_settings(self) -> None:
        # Открываем окно в главном потоке tkinter
        self.root.after(0, lambda: SettingsWindow(self.root, self.ctrl, self))

    def _quit(self) -> None:
        if self._icon:
            self._icon.stop()
        self.root.after(0, self.root.quit)

    def run(self) -> None:
        menu = pystray.Menu(
            pystray.MenuItem("Настройки", lambda icon, item: self._open_settings()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход",     lambda icon, item: self._quit()),
        )
        self._icon = pystray.Icon(
            name="govorun",
            icon=ICON_IDLE,
            title=f"Govorun  [{HOTKEY.upper()}]",
            menu=menu,
        )
        # Трей — в отдельном потоке, tkinter — в главном
        threading.Thread(target=self._icon.run, daemon=True).start()
        self.root.mainloop()


# ============================================================
# Утилиты
# ============================================================
def print_devices() -> None:
    print("\n📋 Доступные устройства ввода:")
    devices   = sd.query_devices()
    default_in = sd.default.device[0]
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " ◄ дефолт" if i == default_in else ""
            print(f"   [{i}] {d['name']}{marker}")
    print()


# ============================================================
# Точка входа
# ============================================================
def main() -> None:
    global INPUT_DEVICE, USE_PUNCT

    load_config()

    parser = argparse.ArgumentParser(description="Голосовой ввод на ПК (офлайн, GigaAM)")
    parser.add_argument("--device",       type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-punct",     action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return

    print_devices()

    if args.device is not None:
        INPUT_DEVICE = args.device
        print(f"🎙  Выбрано устройство [{INPUT_DEVICE}]: {sd.query_devices(INPUT_DEVICE)['name']}\n")
    else:
        default_in = sd.default.device[0]
        dev_name   = sd.query_devices(default_in)["name"] if default_in >= 0 else "неизвестно"
        print(f"🎙  Используется дефолтное устройство [{default_in}]: {dev_name}\n")

    if args.no_punct:
        USE_PUNCT = False
    if USE_PUNCT:
        load_punct_model()

    load_replacements()

    rec  = load_recognizer()
    ctrl = Controller(rec)

    keyboard.add_hotkey(HOTKEY, ctrl.toggle, suppress=True)
    print(f"🎯 [{HOTKEY.upper()}] — старт/стоп записи, [ESC] — отмена.")
    print(f"   Словарь замен: {REPLACEMENTS_PATH}")
    print("   Настройки и выход — иконка в системном трее.\n")

    # Скрытый tkinter root + трей
    root = tk.Tk()
    root.withdraw()

    tray = TrayApp(ctrl, root)
    tray.run()


if __name__ == "__main__":
    main()