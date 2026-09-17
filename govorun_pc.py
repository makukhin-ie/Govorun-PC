#!/usr/bin/env python3
"""
Govorun PC — голосовой ввод на русском для Windows. Полностью офлайн.

Распознавание: GigaAM от Сбера. По умолчанию v3 через onnx-asr — она точнее
и быстрее. Запасной вариант — v2 через sherpa-onnx: работает без интернета
и без лишних пакетов. Модель выбирается в настройках или в config.json.

Пунктуация: xlm-roberta (ONNX).

Использование:
    python govorun_pc.py               # запуск с иконкой в трее
    python govorun_pc.py --device 2    # выбрать микрофон
    python govorun_pc.py --list-devices
    python compare_engines.py --record 40   # сравнить модели на своём голосе

Установка:
    pip install -r requirements.txt
    python download_models.py          # только для запасного движка v2, ~330 МБ

Веса v3 качаются сами при первом запуске (~900 МБ на модель).
"""

from __future__ import annotations

import argparse
import gc
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


def _hub_is_populated() -> bool:
    """Есть ли в кэше хоть одна скачанная модель."""
    try:
        return any((_CACHE / "huggingface" / "hub").glob("models--*"))
    except OSError:
        return False


# Если веса уже лежат в кэше — в сеть при старте не ходим вообще.
# Иначе huggingface_hub на каждом запуске проверяет обновления: это лишняя
# секунда, анонимное предупреждение в логе, а при автозапуске ещё и ожидание
# таймаута, пока Wi-Fi поднимается после входа в систему.
# Понадобится новая модель — офлайн снимется автоматически, см. _with_hub_access.
if _hub_is_populated():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _with_hub_access(load):
    """Повторяет загрузку с доступом в сеть, если офлайн-кэш её не нашёл."""
    try:
        return load()
    except Exception:
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise
        os.environ["HF_HUB_OFFLINE"] = "0"
        try:
            # Библиотека читает переменную один раз при импорте — правим и там
            import huggingface_hub.constants as _hc
            _hc.HF_HUB_OFFLINE = False
        except Exception:
            pass
        print("   В кэше этой модели нет — качаю с Hugging Face.")
        return load()

# Вывод всегда в UTF-8.
#
# При тихом запуске stdout уходит в файл, и Python берёт кодировку системы —
# на русской Windows это cp1251, в которой нет ни одного эмодзи. Любая строка
# со значком роняла программу с UnicodeEncodeError ещё до загрузки модели,
# причём в терминале всё работало: там консоль в UTF-8.
#
# errors="replace" — страховка: если поток не примет UTF-8, вместо падения
# получим вопросительный знак.
#
# line_buffering нужен по той же причине. При выводе в файл Python копит
# строки блоками по 8 КБ и сбрасывает, когда буфер полон или процесс
# завершился. Программа живёт в трее и не завершается, поэтому лог
# оставался пустым именно тогда, когда в него хочется заглянуть.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace",
                            line_buffering=True)
    except (AttributeError, ValueError, OSError):
        pass

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

USE_PUNCT: bool       = True   # пишется в config.json
CONVERT_NUMBERS: bool = True   # числительные словами → цифрами

# Через сколько минут простоя выгружать модели из памяти.
# Программа висит в трее весь день, а веса занимают около двух гигабайт —
# при загруженной оперативке это заметно. Загрузка обратно начинается
# в момент нажатия хоткея, параллельно с записью, поэтому ждать
# не приходится: пока вы говорите, модель успевает подняться.
# 0 — никогда не выгружать (быстрее на первой фразе, дороже по памяти).
IDLE_UNLOAD_MIN: float = 10.0

# Движок распознавания:
#   onnx-v3   — GigaAM v3 через onnx-asr. По умолчанию: точнее и быстрее v2.
#               Веса качаются с Hugging Face при первом запуске (~900 МБ).
#   sherpa-v2 — GigaAM v2 через sherpa-onnx. Запасной вариант: работает
#               без интернета и без onnx-asr, модель кладёт download_models.py
ENGINE: str   = "onnx-v3"
V3_MODEL: str = "gigaam-v3-ctc"

# Что показываем в настройках: (подпись, движок, модель)
ENGINE_CHOICES: list[tuple[str, str, str]] = [
    ("GigaAM v3 CTC — рекомендуется",            "onnx-v3",   "gigaam-v3-ctc"),
    ("GigaAM v3 RNN-T — быстрее, но теряет слова", "onnx-v3", "gigaam-v3-rnnt"),
    ("GigaAM v2 — из комплекта, без установки",  "sherpa-v2", ""),
]


def engine_label(engine: str, model: str) -> str:
    for label, eng, mdl in ENGINE_CHOICES:
        if eng == engine and (eng == "sherpa-v2" or mdl == model):
            return label
    return ENGINE_CHOICES[0][0]


# ============================================================
# Настройки: живут между запусками
# ============================================================
def load_config() -> None:
    """Читает config.json. Отсутствие файла — не ошибка, просто дефолты."""
    global HOTKEY, INPUT_DEVICE, USE_PUNCT, CONVERT_NUMBERS, ENGINE, V3_MODEL
    global IDLE_UNLOAD_MIN
    if not CONFIG_PATH.exists():
        return
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  config.json повреждён, беру значения по умолчанию: {e}")
        return

    HOTKEY = str(data.get("hotkey", HOTKEY)).strip().lower() or HOTKEY
    USE_PUNCT = bool(data.get("punctuation", USE_PUNCT))
    CONVERT_NUMBERS = bool(data.get("convert_numbers", CONVERT_NUMBERS))
    ENGINE   = str(data.get("engine", ENGINE)).strip().lower() or ENGINE
    V3_MODEL = str(data.get("v3_model", V3_MODEL)).strip() or V3_MODEL
    try:
        IDLE_UNLOAD_MIN = max(0.0, float(data.get("idle_unload_min",
                                                  IDLE_UNLOAD_MIN)))
    except (TypeError, ValueError):
        pass

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
                    "convert_numbers": CONVERT_NUMBERS,
                    "engine": ENGINE,
                    "v3_model": V3_MODEL,
                    "idle_unload_min": IDLE_UNLOAD_MIN,
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
class _SherpaV2:
    """GigaAM v2 через sherpa-onnx — исходный движок Говоруна."""

    name = "GigaAM v2 (sherpa-onnx)"

    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not TOKENS_PATH.exists():
            print(f"[!] Модель v2 не найдена в {MODELS_DIR}", file=sys.stderr)
            print("    Либо скачайте её:  python download_models.py",
                  file=sys.stderr)
            print('    Либо поставьте v3:  pip install "onnx-asr[cpu,hub]"',
                  file=sys.stderr)
            sys.exit(1)
        print(f"⏳ Загружаю {self.name}, потоков: {NUM_THREADS}...")
        self._rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(MODEL_PATH),
            tokens=str(TOKENS_PATH),
            num_threads=NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=FEATURE_DIM,
            decoding_method="greedy_search",
        )

    def transcribe(self, audio: np.ndarray) -> str:
        stream = self._rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._rec.decode_stream(stream)
        return stream.result.text.strip()


class _OnnxV3:
    """
    GigaAM v3 через onnx-asr. Модель заметно точнее на русском, но требует
    отдельной установки:  pip install "onnx-asr[cpu,hub]"
    При первом запуске веса качаются с Hugging Face — нужен интернет один раз.
    """

    def __init__(self, model_name: str) -> None:
        self.name = f"GigaAM v3 ({model_name}, onnx-asr)"
        try:
            import onnx_asr
        except ImportError as e:
            raise RuntimeError("пакет onnx-asr не установлен") from e

        # Спрашиваем у onnxruntime, что реально доступно, и берём лучшее.
        # Иначе библиотека просит CUDA вслепую и на каждом старте сыплет
        # предупреждением, что её нет.
        providers = None
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            preferred = [
                p for p in ("CUDAExecutionProvider",      # NVIDIA
                            "DmlExecutionProvider",       # DirectML: Intel, AMD, Qualcomm
                            "CPUExecutionProvider")
                if p in available
            ]
            providers = preferred or None
        except Exception:
            pass

        print(f"⏳ Загружаю {self.name}...")
        if providers:
            print(f"   Вычислитель: {providers[0].replace('ExecutionProvider', '')}")
        print("   Первый запуск качает веса с Hugging Face (~900 МБ), это долго.")
        try:
            self._model = _with_hub_access(
                lambda: onnx_asr.load_model(model_name, providers=providers))
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__}: {e}") from e

    def transcribe(self, audio: np.ndarray) -> str:
        return str(self._model.recognize(audio, sample_rate=SAMPLE_RATE)).strip()


def load_recognizer():
    """
    Собирает движок по настройке engine.
    Если v3 недоступен — не падаем, а откатываемся на v2 из комплекта.
    """
    engine = (ENGINE or "sherpa-v2").strip().lower()

    if engine in ("onnx-v3", "v3", "onnx"):
        try:
            rec = _OnnxV3(V3_MODEL)
            print("✅ Модель готова.\n")
            return rec
        except Exception as e:
            print(f"⚠️  Движок v3 недоступен: {e}")
            print("   Работаю на GigaAM v2 из комплекта.")
            print('   Чтобы включить v3:  pip install "onnx-asr[cpu,hub]"\n')
    elif engine != "sherpa-v2":
        print(f"⚠️  Неизвестный движок «{engine}», беру v2")

    rec = _SherpaV2()
    print("✅ Модель готова.\n")
    return rec


def _recognize_one(rec, audio: np.ndarray) -> str:
    """Распознать один кусок, который заведомо помещается в контекст модели."""
    if audio.size / SAMPLE_RATE < MIN_DURATION_SEC:
        return ""
    return rec.transcribe(audio)


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
    rec,
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
# Модель не ставится / не грузится — не пытаться снова на каждой фразе
_punct_broken = False


def unload_punct_model() -> None:
    """Освобождает память под пунктуатором. Загрузится заново, когда нужен."""
    global _punct_model
    if _punct_model is None:
        return
    _punct_model = None
    gc.collect()


def load_punct_model() -> bool:
    global _punct_model, _punct_broken
    if _punct_model is not None:
        return True
    if _punct_broken:
        return False
    try:
        import warnings

        from punctuators.models import PunctCapSegModelONNX
        print("⏳ Загружаю модель пунктуации (ONNX)...")
        # Пунктуатор просит CUDA не спрашивая и на машинах без неё сыплет
        # предупреждением при каждом старте. Работать это не мешает —
        # он молча уходит на процессор, — но лог замусоривает.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not in available provider names.*")
            _punct_model = _with_hub_access(
                lambda: PunctCapSegModelONNX.from_pretrained(
                    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
                ))
        print("✅ Пунктуация готова.\n")
        return True
    except ImportError:
        print("⚠️  punctuators не установлен — пунктуация отключена.\n")
        _punct_broken = True
        return False
    except Exception as e:
        print(f"⚠️  Не удалось загрузить модель пунктуации: {e}\n")
        _punct_broken = True
        return False


def restore_punctuation(text: str) -> str:
    if not text:
        return text
    # Модель могли выгрузить по простою — поднимаем обратно
    if USE_PUNCT and _punct_model is None:
        load_punct_model()
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
# Числа словами → цифрами
# ============================================================
# «двадцать четыре» → «24», «пятнадцатого сентября» → «15 сентября»,
# «в две тысячи двадцать шестом году» → «в 2026 году».
#
# Правило намеренно осторожное: одиночные числительные от нуля до десяти
# остаются словами. «Один момент» не должен превращаться в «1 момент»,
# а «во-первых» — тем более. Цифрами пишем то, что словами читается плохо:
# составные числительные и всё, что больше десяти.

_NUM_UNITS = {
    "ноль": 0, "нуль": 0,
    "один": 1, "одна": 1, "одно": 1, "одни": 1,
    "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
}
_NUM_TEENS = {
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_NUM_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_NUM_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700,
    "восемьсот": 800, "девятьсот": 900,
}
_NUM_SCALES = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "тысячу": 1000,
    "миллион": 10**6, "миллиона": 10**6, "миллионов": 10**6,
    "миллиард": 10**9, "миллиарда": 10**9, "миллиардов": 10**9,
}

_NUM_SIMPLE = {**_NUM_UNITS, **_NUM_TEENS, **_NUM_TENS, **_NUM_HUNDREDS}
_NUM_ALL    = {**_NUM_SIMPLE, **_NUM_SCALES}

# Порядковые: основа → значение. Окончания перебираем отдельно, чтобы
# ловить все падежи: «пятого», «пятом», «пятый», «пятая».
_ORD_STEMS = {
    "перв": 1, "втор": 2, "четвёрт": 4, "четверт": 4, "пят": 5, "шест": 6,
    "седьм": 7, "восьм": 8, "девят": 9, "десят": 10, "одиннадцат": 11,
    "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14, "пятнадцат": 15,
    "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19,
    "двадцат": 20, "тридцат": 30, "сороков": 40, "пятидесят": 50,
    "шестидесят": 60, "семидесят": 70, "восьмидесят": 80, "девяност": 90,
    "сот": 100, "тысячн": 1000,
}
_ORD_ENDINGS = ("ый", "ой", "ий", "ого", "его", "ом", "ем", "ому", "ему",
                "ым", "им", "ая", "яя", "ую", "юю", "ое", "ее", "ые", "ие",
                "ых", "их", "ыми", "ими")

_ORDINALS: dict[str, int] = {}
for _stem, _val in _ORD_STEMS.items():
    for _end in _ORD_ENDINGS:
        _ORDINALS.setdefault(_stem + _end, _val)
# «третий» выпадает из схемы — вписываем руками
for _f in ("третий", "третьего", "третьем", "третья", "третьей", "третье",
           "третьи", "третьих", "трети"):
    _ORDINALS[_f] = 3

_MONTHS = {
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
}
_YEAR_WORDS = {"год", "года", "году", "годе", "годом"}

_WORD_RE = re.compile(r"[а-яёА-ЯЁ]+")

GROUP_FROM = 10_000   # с какого числа ставим разряды: 40 000, но 1500


def _fmt(value: int) -> str:
    if value >= GROUP_FROM:
        return f"{value:,}".replace(",", " ")
    return str(value)


def _parse_run(words: list[str]) -> int | None:
    """Считает значение цепочки числительных. None — если цепочка бессмысленная."""
    total, current, seen = 0, 0, False
    for w in words:
        if w in _NUM_SCALES:
            scale = _NUM_SCALES[w]
            total += (current or 1) * scale
            current = 0
            seen = True
        elif w in _NUM_SIMPLE:
            current += _NUM_SIMPLE[w]
            seen = True
        else:
            return None
    return total + current if seen else None


def convert_numbers(text: str) -> str:
    """Заменяет числительные словами на цифры там, где это уместно."""
    if not text:
        return text

    tokens = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if not tokens:
        return text

    replacements: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        word, start, _ = tokens[i]
        low = word.lower()

        if low not in _NUM_ALL:
            i += 1
            continue

        # Собираем максимальную цепочку числительных
        j = i
        while j + 1 < len(tokens) and tokens[j + 1][0].lower() in _NUM_ALL:
            # Между словами должны быть только пробелы или дефис
            between = text[tokens[j][2]:tokens[j + 1][1]]
            if between.strip(" - "):
                break
            j += 1

        chain = [t[0].lower() for t in tokens[i:j + 1]]
        end   = tokens[j][2]
        value = _parse_run(chain)

        if value is None:
            i = j + 1
            continue

        nxt      = tokens[j + 1][0].lower() if j + 1 < len(tokens) else ""
        ordinal  = _ORDINALS.get(nxt)
        is_scale = len(chain) == 1 and chain[0] in _NUM_SCALES

        # «две тысячи двадцать шестого года» → «2026 года»
        after_ord = tokens[j + 2][0].lower() if j + 2 < len(tokens) else ""
        if ordinal is not None and after_ord in _YEAR_WORDS:
            replacements.append((start, tokens[j + 1][2], _fmt(value + ordinal)))
            i = j + 2
            continue

        # «двадцать пятого сентября» → «25 сентября»
        if (ordinal is not None and after_ord in _MONTHS
                and 1 <= value + ordinal <= 31):
            replacements.append((start, tokens[j + 1][2], str(value + ordinal)))
            i = j + 2
            continue

        if len(chain) >= 2 or (value >= 11 and not is_scale):
            replacements.append((start, end, _fmt(value)))
        i = j + 1

    # Отдельный проход: даты вида «пятнадцатого сентября»
    for k, (word, start, end) in enumerate(tokens):
        low = word.lower()
        val = _ORDINALS.get(low)
        if val is None or not (1 <= val <= 31):
            continue
        nxt = tokens[k + 1][0].lower() if k + 1 < len(tokens) else ""
        if nxt not in _MONTHS:
            continue
        # Составные даты вроде «двадцать пятого сентября» уже собрал
        # основной проход — пересекающиеся замены пропускаем
        if any(s < end and start < e for s, e, _ in replacements):
            continue
        replacements.append((start, end, str(val)))

    for start, end, new in sorted(replacements, reverse=True):
        text = text[:start] + new + text[end:]
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
    def __init__(self, rec):
        self.rec      = rec
        self.recorder = Recorder()
        self._busy      = False
        self._busy_lock = threading.Lock()
        # Загрузка модели: чтобы две попытки не пошли параллельно
        self._rec_lock  = threading.Lock()
        self._last_use  = time.monotonic()
        self.on_state_change: callable = lambda recording: None  # колбэк для трея
        # (номер фрагмента, всего). (0, 0) — обработка закончена
        self.on_progress: callable = lambda done, total: None
        # Esc нажат: распознаём, но не вставляем. Трей помечает бейдж
        self.on_cancel_start: callable = lambda: None
        self.on_cancel_done:  callable = lambda text: None
        self._paste_result = True
        self._esc_hook = None

    # --- жизненный цикл модели ------------------------------------------
    def ensure_engine(self):
        """
        Возвращает движок, подняв его, если он был выгружен по простою.
        Блокировка нужна потому, что вызывается из двух мест сразу:
        из фонового прогрева при старте записи и из обработки.
        """
        with self._rec_lock:
            if self.rec is None:
                print("⏳ Поднимаю модель из кэша...")
                self.rec = load_recognizer()
            self._last_use = time.monotonic()
            return self.rec

    def unload_engine(self) -> bool:
        """Выгружает модели, если сейчас ничего не происходит."""
        with self._busy_lock:
            if self._busy:
                return False
        if self.recorder.recording:
            return False
        with self._rec_lock:
            if self.rec is None:
                return False
            self.rec = None
        unload_punct_model()
        gc.collect()
        print(f"💤 Простой {IDLE_UNLOAD_MIN:.0f} мин — модели выгружены, "
              f"память освобождена.\n")
        return True

    def start_idle_watch(self) -> None:
        """Сторож простоя. Проверяет раз в полминуты, спит в фоне."""
        if IDLE_UNLOAD_MIN <= 0:
            return

        def watch() -> None:
            while True:
                time.sleep(30)
                if IDLE_UNLOAD_MIN <= 0:
                    continue
                if self.rec is None:
                    continue
                if time.monotonic() - self._last_use < IDLE_UNLOAD_MIN * 60:
                    continue
                try:
                    self.unload_engine()
                except Exception as e:
                    print(f"[!] Не удалось выгрузить модель: {e}")

        threading.Thread(target=watch, daemon=True).start()

    def reload_engine(self) -> None:
        """
        Меняет модель на лету. Грузится в фоне — иначе окно настроек
        зависнет на несколько секунд. На время загрузки запись заблокирована.
        """
        with self._busy_lock:
            if self._busy:
                print("[!] Идёт обработка записи, смена модели отложена")
                return
            self._busy = True

        def work() -> None:
            try:
                with self._rec_lock:
                    self.rec = load_recognizer()
                    self._last_use = time.monotonic()
            except SystemExit:
                print("[!] Новая модель не загрузилась, остаётся прежняя\n")
            except Exception as e:
                print(f"[!] Новая модель не загрузилась ({e}), остаётся прежняя\n")
            finally:
                with self._busy_lock:
                    self._busy = False

        threading.Thread(target=work, daemon=True).start()

    def toggle(self, _event=None) -> None:
        with self._busy_lock:
            if self._busy:
                return
        if self.recorder.recording:
            self._finish(paste=True)
        else:
            print(f"🎤 Запись... ({HOTKEY.upper()} — стоп, ESC — отмена)")
            self.recorder.start()
            self._hook_esc()
            self.on_state_change(True)
            # Прогрев: если модель выгружена по простою, поднимаем её прямо
            # сейчас, пока человек говорит. К моменту остановки записи она
            # готова, и выгрузка ничего не стоит по времени.
            if self.rec is None:
                threading.Thread(target=self.ensure_engine, daemon=True).start()

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
        """
        Esc отменяет ВСТАВКУ, а не запись.

        Раньше он выбрасывал записанное — и одно случайное нажатие стирало
        несколько минут речи безвозвратно. Теперь запись всё равно
        распознаётся, текст уходит в буфер и в last.txt, не происходит только
        вставка в активное окно. Отмена в реальности почти всегда значит
        «не сюда», а не «сотри навсегда»; а если запись и правда не нужна —
        лежащий в файле текст ничего не стоит.
        """
        if not self.recorder.recording:
            return
        print("✖  Вставка отменена — текст сохраню в буфер и в last.txt")
        self.on_cancel_start()
        self._finish(paste=False)

    def _finish(self, paste: bool) -> None:
        """Остановить запись и отправить её в обработку."""
        with self._busy_lock:
            if self._busy:
                return
            self._busy = True
        self._unhook_esc()
        self._paste_result = paste
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self) -> None:
        paste = self._paste_result
        self._paste_result = True          # следующая запись — обычная
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
            text = recognize(self.ensure_engine(), audio, on_progress=progress)
        except Exception as e:
            print(f"[!] Ошибка распознавания: {e}")
            text = ""

        self.on_progress(0, 0)

        if text:
            # Числа — ДО пунктуации. Иначе модель пунктуации успевает
            # вставить запятую внутрь числительного («триста, шестьдесят
            # пять»), и оно распадается на «300, 65». Замены — после,
            # чтобы truecase не трогал уже готовую латиницу.
            if CONVERT_NUMBERS:
                text = convert_numbers(text)
            text = restore_punctuation(text)
            text = apply_replacements(text)
            # Сохраняем ДО вставки — тогда даже падение на вставке не съест текст
            save_fallback(text)
            print(f"📝 {text}")
            if not paste:
                # Вставку отменили — но текст всё равно кладём в буфер,
                # чтобы Ctrl+V сработал там, где человек решит
                try:
                    pyperclip.copy(text)
                except Exception as e:
                    print(f"[!] Буфер обмена недоступен: {e}")
                print(f"✖  Вставка отменена, текст в буфере — Ctrl+V куда нужно.")
                print(f"   Копия: {FALLBACK_DIR / 'last.txt'}\n")
                self.on_cancel_done(text)
            elif paste_text(text):
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

        # --- Модель распознавания ---
        tk.Label(f, text="Модель:", anchor="w").grid(
            row=2, column=0, sticky="w", **pad)
        self._engine_labels = [c[0] for c in ENGINE_CHOICES]
        self._engine_var = tk.StringVar(value=engine_label(ENGINE, V3_MODEL))
        ttk.Combobox(
            f,
            textvariable=self._engine_var,
            values=self._engine_labels,
            state="readonly",
            width=30,
        ).grid(row=2, column=1, sticky="ew", **pad)

        tk.Label(
            f,
            text="v3 точнее и быстрее, но требует onnx-asr\n"
                 "и качает ~900 МБ при первом включении",
            anchor="w", justify="left", fg="#666",
        ).grid(row=3, columnspan=2, sticky="w", padx=10)

        # --- Пунктуация и числа ---
        self._punct_var = tk.BooleanVar(value=USE_PUNCT)
        tk.Checkbutton(
            f, text="Восстанавливать пунктуацию", variable=self._punct_var
        ).grid(row=4, columnspan=2, sticky="w", **pad)

        self._numbers_var = tk.BooleanVar(value=CONVERT_NUMBERS)
        tk.Checkbutton(
            f, text="Числа цифрами: «двадцать четыре» → «24»",
            variable=self._numbers_var
        ).grid(row=5, columnspan=2, sticky="w", **pad)

        # --- Словарь замен ---
        tk.Button(
            f, text="Открыть словарь замен…", width=24,
            command=self._open_replacements,
        ).grid(row=6, columnspan=2, sticky="w", **pad)

        # --- Кнопки ---
        btn_frame = tk.Frame(f)
        btn_frame.grid(row=7, columnspan=2, pady=(8, 0))
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

        # Пунктуация и числа
        global USE_PUNCT, CONVERT_NUMBERS
        want_punct = self._punct_var.get()
        USE_PUNCT = want_punct
        if want_punct:
            load_punct_model()      # уже загружена — вернётся сразу
        else:
            unload_punct_model()

        CONVERT_NUMBERS = self._numbers_var.get()

        # Модель. Меняется редко, зато перезагрузка долгая — делаем в фоне
        global ENGINE, V3_MODEL
        chosen = self._engine_var.get()
        for label, eng, mdl in ENGINE_CHOICES:
            if label != chosen:
                continue
            if eng != ENGINE or (mdl and mdl != V3_MODEL):
                ENGINE = eng
                V3_MODEL = mdl or V3_MODEL
                print(f"🔄 Переключаюсь на: {label}")
                self.ctrl.reload_engine()
            break

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
        # Плашка-подсказка живёт отдельно от бейджа записи
        self._notice_win: tk.Toplevel | None = None
        self._notice_job = None
        # Откуда брать громкость микрофона; ставится снаружи
        self.level_provider: callable = lambda: 0.0

    # --- плашка-подсказка ------------------------------------------------
    NOTICE_W, NOTICE_H = 330, 62

    def notice(self, title: str, subtitle: str, ms: int = 7000) -> None:
        """
        Плашка в том же углу, где бейдж записи, и того же вида — только шире
        и в две строки.

        Сделана своим окном, а не уведомлением Windows: системные
        уведомления глушит фокус-помощник и настройки приложения, они
        оседают в центре уведомлений и до человека не доходят. Бейдж же
        виден всегда — значит и подсказку надо показывать там же.

        Держится ms миллисекунд, чтобы успеть прочитать, и убирается
        по клику, если мешает.
        """
        self._close_notice()
        W, H = self.NOTICE_W, self.NOTICE_H
        try:
            win = tk.Toplevel(self._root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            win.attributes("-alpha", 0.94)
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"{W}x{H}+{sw - W - 16}+{sh - H - 52}")

            c = tk.Canvas(win, width=W, height=H, bg=self.BG, highlightthickness=0)
            c.pack()
            r = 10
            c.create_arc(0, 0, 2*r, 2*r, start=90, extent=90,
                         fill=self.BG, outline=self.BG)
            c.create_arc(W-2*r, 0, W, 2*r, start=0, extent=90,
                         fill=self.BG, outline=self.BG)
            c.create_rectangle(r, 0, W-r, H, fill=self.BG, outline=self.BG)
            c.create_rectangle(0, r, W, H-r, fill=self.BG, outline=self.BG)

            c.create_oval(14, H//2 - 5, 24, H//2 + 5,
                          fill=self.BAR_FG, outline="")
            c.create_text(36, 20, anchor="w", text=title, fill=self.FG,
                          font=("Segoe UI", 10, "bold"))
            c.create_text(36, 40, anchor="w", text=subtitle, fill="#c8c8c8",
                          font=("Segoe UI", 9))

            # Клик убирает плашку, не дожидаясь таймера
            for w in (win, c):
                w.bind("<Button-1>", lambda _e: self._close_notice())

            self._notice_win = win
            self._notice_job = self._root.after(ms, self._close_notice)
        except Exception as e:
            print(f"[!] Не удалось показать подсказку: {e}")

    def _close_notice(self) -> None:
        if self._notice_job:
            try:
                self._root.after_cancel(self._notice_job)
            except Exception:
                pass
            self._notice_job = None
        if self._notice_win is not None:
            try:
                if self._notice_win.winfo_exists():
                    self._notice_win.destroy()
            except Exception:
                pass
            self._notice_win = None

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
        self._cancelled = False
        ctrl.on_state_change = self._on_state_change
        ctrl.on_progress     = self._on_progress
        ctrl.on_cancel_start = self._on_cancel_start
        ctrl.on_cancel_done  = self._on_cancel_done
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
            # чтобы не стоять и не гадать, работает оно или зависло.
            # После Esc — с крестиком: считается, но не вставится.
            mark = "x" if self._cancelled else "⏳"
            self.root.after(0, lambda: self.overlay.show(f"{mark} ..."))

    def _on_cancel_start(self) -> None:
        """Esc нажат: бейдж продолжает считать фрагменты, но помечен крестиком."""
        self._cancelled = True
        self.root.after(0, lambda: self.overlay.show("x ..."))

    def _on_cancel_done(self, text: str) -> None:
        """Обработка кончилась: сказать, что текст не пропал, а лежит в буфере."""
        self._cancelled = False
        self.root.after(0, lambda: self.overlay.notice(
            "Вставка отменена",
            "Текст в буфере — вставьте через Ctrl+V",
        ))

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self._cancelled = False
            self.root.after(0, self.overlay.hide)
        else:
            # Крестик вместо песочных часов — видно, что считается,
            # но вставки не будет
            mark = "x" if self._cancelled else "⏳"
            self.root.after(0, lambda: self.overlay.set_label(f"{mark} {done}/{total}"))

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
# Одна копия за раз
# ============================================================
# При автозапуске легко получить две копии: одну поднял вход в систему,
# вторую запустили руками. Обе вешают один и тот же хоткей с suppress,
# и поведение становится непредсказуемым.
#
# Замок держим на сокете, а не на файле: если процесс убили, порт
# освобождается сам. Файловый замок пришлось бы чистить руками.
SINGLE_INSTANCE_PORT = 47821
_instance_lock = None


def ensure_single_instance() -> None:
    global _instance_lock
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", SINGLE_INSTANCE_PORT))
        sock.listen(1)
    except OSError:
        print("[!] Govorun уже запущен — вторая копия не нужна.", file=sys.stderr)
        print("    Иконка в системном трее, рядом с часами.", file=sys.stderr)
        sys.exit(1)
    _instance_lock = sock   # держим ссылку, иначе сокет закроется


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

    ensure_single_instance()

    print(f"🕘 Старт: {time.strftime('%Y-%m-%d %H:%M:%S')}")
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

    ctrl.start_idle_watch()

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