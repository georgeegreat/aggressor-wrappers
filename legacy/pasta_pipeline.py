#!/usr/bin/env python3
"""
Единый пайплайн: PASTA 2.0 → распаковка batch.tar → сравнение профилей агрегации.

Этапы:
  1. pasta   — отправка FASTA на PASTA 2.0, скачивание batch.tar
  2. extract — распаковка архива, извлечение *.seq.aggr_profile.dat (PASTA)
  3. analyze — сравнение мутантов с референсом, графики и CSV
  4. full    — все этапы подряд

Зависимости:
  pip install biopython selenium numpy matplotlib pandas

Для этапа pasta нужен Microsoft Edge; драйвер подбирается автоматически (Selenium Manager).
Опционально: --driver путь к msedgedriver.exe, если версия совпадает с браузером.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from matplotlib.backends.backend_pdf import PdfPages
from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service as EdgeService

PASTA_URL = "http://old.protein.bio.unipd.it/pasta2/"
DEFAULT_WAIT_TIMEOUT = 3600  # 1 час — PASTA может считать долго
POLL_INTERVAL_SEC = 5


# PASTA 2.0 — отправка FASTA

def read_fasta_records(fasta_file: str) -> List:
    """Читает все записи из FASTA."""
    try:
        return list(SeqIO.parse(fasta_file, "fasta"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Файл не найден: {fasta_file}") from exc


def build_combined_fasta(records: List) -> Tuple[str, int]:
    """Собирает мульти-FASTA строку и суммарную длину."""
    combined = ""
    total_length = 0
    for record in records:
        combined += f">{record.description}\n{record.seq}\n\n"
        total_length += len(record.seq)
    return combined, total_length


def create_edge_driver(
    driver_path: Optional[str] = None,
    headless: bool = True,
) -> webdriver.Edge:
    """
    Создаёт Edge WebDriver.

    По умолчанию Selenium Manager сам скачивает msedgedriver под версию браузера.
    Если указан --driver и версия не совпадает — повторная попытка без него.
    """
    options = webdriver.EdgeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    if driver_path:
        try:
            print(f"Запуск Edge с драйвером: {driver_path}")
            return webdriver.Edge(
                service=EdgeService(driver_path),
                options=options,
            )
        except SessionNotCreatedException as exc:
            if "only supports Microsoft Edge version" in str(exc):
                print(
                    "\nВерсия msedgedriver не совпадает с Edge. "
                    "Пробую автоматический подбор драйвера (Selenium Manager)..."
                )
            else:
                raise

    print("Запуск Edge (автоподбор драйвера через Selenium Manager)...")
    return webdriver.Edge(options=options)


def find_tar_download_link(driver: webdriver.Edge) -> Tuple[Optional[str], Optional[str]]:
    """
    Ищет ссылку на batch.tar / batch.tar.gz на странице.

    Сначала парсит HTML (устойчиво к StaleElementReference при перерисовке страницы),
    затем при необходимости обходит <a> с повторными попытками.
    """
    try:
        html = driver.page_source
    except StaleElementReferenceException:
        return None, None

    patterns = [
        r'href=["\']([^"\']*batch\.tar(?:\.gz)?[^"\']*)["\']',
        r'(https?://[^\s"\'<>]+/batch\.tar(?:\.gz)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.I)
        if match:
            return match.group(1), "batch.tar"

    for attempt in range(3):
        try:
            for link in driver.find_elements(By.TAG_NAME, "a"):
                try:
                    href = link.get_attribute("href") or ""
                    text = (link.text or "").strip()
                except StaleElementReferenceException:
                    continue
                if not href:
                    continue
                if "batch.tar" in href:
                    return href, text or "batch.tar"
                if ("batch" in href.lower() or ".tar" in href.lower()) and not href.endswith(".html"):
                    return href, text or "архив"
            break
        except StaleElementReferenceException:
            time.sleep(0.5)

    return None, None


def wait_for_pasta_results(
    driver: webdriver.Edge,
    timeout_sec: int = DEFAULT_WAIT_TIMEOUT,
    poll_interval: int = POLL_INTERVAL_SEC,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ждёт появления ссылки на batch.tar (опрос страницы, без фиксированного сна).
    """
    print(f"\nОжидание результатов PASTA (до {timeout_sec // 60} мин, опрос каждые {poll_interval} с)...")
    start = time.time()
    last_status = -1

    while time.time() - start < timeout_sec:
        elapsed = int(time.time() - start)

        try:
            tar_url, tar_text = find_tar_download_link(driver)
            if tar_url:
                print(f"  Готово за {elapsed} с.")
                return tar_url, tar_text

            page_lower = driver.page_source.lower()
            if any(err in page_lower for err in ("fatal error", "query failed", "too many sequences")):
                print("  PASTA вернул сообщение об ошибке на странице.")
                return None, None

            if elapsed // 30 != last_status:
                last_status = elapsed // 30
                try:
                    n_links = len(driver.find_elements(By.TAG_NAME, "a"))
                except StaleElementReferenceException:
                    n_links = "?"
                url = driver.current_url[:70]
                print(f"  {elapsed} с... URL: {url} | ссылок: {n_links}")

        except StaleElementReferenceException:
            if elapsed // 30 != last_status:
                last_status = elapsed // 30
                print(f"  {elapsed} с... страница обновляется, жду...")

        time.sleep(poll_interval)

    print(f"  Таймаут {timeout_sec} с — ссылка на batch.tar не появилась.")
    return None, None


def submit_all_fasta_at_once(
    fasta_file: str,
    driver_path: Optional[str] = None,
    save_dir: str = "PASTA_Results",
    headless: bool = True,
    max_sequences: Optional[int] = None,
    wait_timeout: int = DEFAULT_WAIT_TIMEOUT,
) -> Optional[str]:
    """
    Отправляет все FASTA-последовательности в PASTA 2.0 одной формой.
    Возвращает путь к скачанному batch.tar или None.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Чтение {fasta_file}...")
    records = read_fasta_records(fasta_file)
    print(f"Найдено {len(records)} последовательностей")

    if max_sequences and len(records) > max_sequences:
        print(
            f"\nВнимание: в файле {len(records)} последовательностей, "
            f"будут отправлены первые {max_sequences} (--max-sequences). "
            "PASTA и браузер могут не справиться с сотнями записей за раз."
        )
        records = records[:max_sequences]

    if len(records) > 100:
        print(
            f"\nПредупреждение: {len(records)} последовательностей — очень много для одной "
            "отправки в PASTA. Рекомендуется разбить FASTA на части или использовать --max-sequences."
        )

    combined_fasta, total_length = build_combined_fasta(records)
    for i, record in enumerate(records, 1):
        print(f"  {i}. {record.id}: {len(record.seq)} аа")
    print(f"\nИтого: {len(records)} белков, {total_length} аминокислот")

    driver = create_edge_driver(driver_path=driver_path, headless=headless)

    try:
        print("\nОткрываю PASTA 2.0...")
        driver.get(PASTA_URL)
        time.sleep(3)

        textarea = driver.find_element(By.ID, "sequence")
        print(f"Ввожу {len(records)} последовательностей...")
        textarea.clear()
        # JS быстрее send_keys для больших FASTA
        driver.execute_script(
            "arguments[0].value = arguments[1];",
            textarea,
            combined_fasta,
        )

        submit = driver.find_element(By.CSS_SELECTOR, "input[name='Submit Query']")
        print("Отправляю запрос в PASTA...")
        submit.click()

        tar_url, tar_text = wait_for_pasta_results(driver, timeout_sec=wait_timeout)

        if not tar_url:
            print("\nbatch.tar не найден. Первые ссылки на странице:")
            try:
                html = driver.page_source
                for i, m in enumerate(re.finditer(r'href=["\']([^"\']+)["\']', html)):
                    if i >= 20:
                        break
                    href = m.group(1)
                    if "tar" in href.lower() or "download" in href.lower():
                        print(f"  {i + 1}. -> {href[:80]}...")
            except Exception:
                pass
            return None

        print(f"Найден: {tar_text}")
        print(f"URL: {tar_url}")

        base_name = os.path.splitext(os.path.basename(fasta_file))[0]
        filename = f"{base_name}_ALL_batch.tar"
        if ".gz" in tar_url:
            filename += ".gz"

        filepath = os.path.join(save_dir, filename)
        print(f"Скачиваю {filename}...")
        urllib.request.urlretrieve(tar_url, filepath)

        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) // 1024
            print(f"Скачан: {filename} ({size_kb} KB)")
            if size_kb < 10:
                print("Внимание: архив очень маленький, возможно пустой")
            return filepath

        print("Ошибка: файл не скачан")
        return None

    except Exception as exc:
        print(f"Ошибка: {type(exc).__name__}: {exc}")
        return None
    finally:
        driver.quit()



# Распаковка batch.tar
def _is_numeric_profile(path: Path) -> bool:
    """Проверяет, что файл — числовой профиль (одно значение на строку)."""
    try:
        data = np.loadtxt(path)
        return data.ndim == 1 and len(data) >= 5
    except (ValueError, OSError):
        return False


def pasta_dat_to_short_name(dat_path: Path) -> str:
    """
    PASTA: RPS2_human_W60Y...fasta.seq.aggr_profile.dat -> RPS2_human_W60Y.txt
    """
    name = dat_path.name
    if ".fasta.seq.aggr_profile.dat" in name:
        base = name.split(".fasta.seq.aggr_profile.dat")[0]
    elif name.endswith(".seq.aggr_profile.dat"):
        base = name[: -len(".seq.aggr_profile.dat")]
    else:
        base = dat_path.stem

    if "MERGEDRULES" in base:
        base = base.split("MERGEDRULES")[0]
    return f"{base}.txt"


def _find_pasta_aggr_profile_files(root: Path) -> List[Path]:
    """Находит файлы агрегационного профиля PASTA (без .free_energy дубликатов)."""
    candidates = list(root.rglob("*.seq.aggr_profile.dat"))
    profiles = []
    for path in candidates:
        if path.name.endswith(".free_energy"):
            continue
        if ".aggr_profile.dat.free_energy" in path.name:
            continue
        if _is_numeric_profile(path):
            profiles.append(path)
    return profiles


def extract_tar_profiles(
    tar_path: str,
    extract_dir: Optional[str] = None,
) -> Path:
    """
    Распаковывает batch.tar.gz и извлекает профили агрегации PASTA.

    PASTA кладёт данные в predictions/*.fasta.seq.aggr_profile.dat
    Скрипт копирует их в profiles/ с короткими именами (*.txt) для анализа.
    """
    tar_path = str(tar_path)
    if extract_dir is None:
        extract_dir = os.path.splitext(tar_path)[0]
        if extract_dir.endswith(".tar"):
            extract_dir = extract_dir[:-4]
        extract_dir += "_extracted"

    out_dir = Path(extract_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tar_path.endswith((".tar.gz", ".tgz")):
        mode = "r:gz"
    elif tar_path.endswith(".tar"):
        mode = "r"
    else:
        raise ValueError(f"Неизвестный формат архива: {tar_path}")

    print(f"\nРаспаковываю {os.path.basename(tar_path)} -> {out_dir}")
    with tarfile.open(tar_path, mode) as archive:
        archive.extractall(out_dir)

    pasta_profiles = _find_pasta_aggr_profile_files(out_dir)
    txt_legacy = [p for p in out_dir.rglob("*.txt") if _is_numeric_profile(p)]
    print(f"  PASTA .aggr_profile.dat: {len(pasta_profiles)}, legacy .txt: {len(txt_legacy)}")

    profiles_dir = out_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    seen_names: set = set()
    for src in pasta_profiles:
        short_name = pasta_dat_to_short_name(src)
        if short_name in seen_names:
            continue
        seen_names.add(short_name)
        dst = profiles_dir / short_name
        dst.write_bytes(src.read_bytes())

    for src in txt_legacy:
        dst = profiles_dir / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    n_out = len(list(profiles_dir.glob("*.txt")))
    print(f"  Профилей в {profiles_dir}: {n_out}")
    for p in sorted(profiles_dir.glob("*.txt")):
        print(f"    - {p.name}")

    if n_out == 0:
        raise FileNotFoundError(
            "В архиве не найдены профили агрегации (*.seq.aggr_profile.dat). "
            "Проверьте содержимое predictions/ в распакованном каталоге."
        )

    return profiles_dir


def resolve_reference_file(
    profiles_dir: Path,
    reference_file: Optional[str] = None,
    reference_id: Optional[str] = None,
    reference_index: Optional[int] = None,
    fasta_file: Optional[str] = None,
) -> Path:
    """
    Определяет файл референса среди профилей PASTA.

    Приоритет:
      1. --reference-file (явный путь)
      2. --reference-id (подстрока в имени файла)
      3. --reference-index (N-я запись FASTA, 0-based)
      4. первая запись FASTA
      5. файл без паттерна мутации A123B
    """
    if reference_file:
        ref = Path(reference_file)
        if not ref.exists():
            raise FileNotFoundError(f"Референс не найден: {reference_file}")
        return ref

    profile_files = sorted(profiles_dir.glob("*.txt"))
    if not profile_files:
        raise FileNotFoundError(f"Нет .txt профилей в {profiles_dir}")

    if reference_id:
        matches = [p for p in profile_files if reference_id.lower() in p.stem.lower()]
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Референс с id '{reference_id}' не найден в {profiles_dir}")

    if fasta_file:
        records = read_fasta_records(fasta_file)
        idx = reference_index if reference_index is not None else 0
        if idx < len(records):
            ref_id = records[idx].id
            exact = [p for p in profile_files if p.stem == ref_id]
            if exact:
                return exact[0]
            for p in profile_files:
                if p.stem.startswith(ref_id) and extract_mutation_info(p.name)[1] is None:
                    return p
            for p in profile_files:
                if ref_id in p.stem and extract_mutation_info(p.name)[1] is None:
                    return p

    non_mutant = [p for p in profile_files if extract_mutation_info(p.name)[1] is None]
    if len(non_mutant) == 1:
        return non_mutant[0]

    print(
        "Предупреждение: референс не определён однозначно, "
        f"используется первый файл: {profile_files[0].name}"
    )
    return profile_files[0]



# Анализ профилей (сравнение с референсом)
def find_optimal_window(
    diff_data: np.ndarray,
    mutation_pos: int,
    max_window_size: int = 50,
    threshold_factor: float = 0.3,
) -> Tuple[slice, int, float]:
    """Находит оптимальное окно анализа вокруг мутации."""
    center = mutation_pos
    n = len(diff_data)

    search_radius = min(max_window_size // 2, center, n - center - 1)
    search_slice = slice(
        max(0, center - search_radius),
        min(n, center + search_radius + 1),
    )

    peak_idx = int(np.argmax(np.abs(diff_data[search_slice])))
    peak_pos = search_slice.start + peak_idx
    peak_val = float(diff_data[peak_pos])

    threshold = abs(peak_val) * threshold_factor

    left = peak_pos
    while left > max(0, peak_pos - max_window_size):
        if abs(diff_data[left - 1]) < threshold:
            break
        left -= 1

    right = peak_pos
    while right < min(n - 1, peak_pos + max_window_size):
        if abs(diff_data[right + 1]) < threshold:
            break
        right += 1

    left = max(0, (left // 5) * 5)
    right = min(n - 1, ((right // 5) * 5) + 5)

    if right - left < 10:
        left = max(0, center - 5)
        right = min(n - 1, center + 5)
        if right - left < 10:
            right = min(n - 1, left + 10)

    return slice(left, right + 1), peak_pos, peak_val


def extract_mutation_info(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str], str]:
    """Извлекает WT, позицию, MUT из имени файла (паттерн _W60Y или W60Y)."""
    name = Path(filename).stem
    match = re.search(r"_([A-Z])(\d+)([A-Z])(?:MERGED|_|$)", name)
    if not match:
        match = re.search(r"([A-Z])(\d+)([A-Z])", name)
    if match:
        wt_aa = match.group(1)
        pos = int(match.group(2))
        mut_aa = match.group(3)
        return wt_aa, pos, mut_aa, f"{wt_aa}{pos}{mut_aa}"
    return None, None, None, name


def create_comparison_plot(
    mutant_data: np.ndarray,
    ref_data: np.ndarray,
    difference: np.ndarray,
    analysis_slice: slice,
    mut_code: str,
    mutant_name: str,
    wt_aa: Optional[str],
    mut_pos: Optional[int],
    mut_aa: Optional[str],
    peak_pos: int,
    peak_val: float,
) -> plt.Figure:
    """Строит 4-панельный график сравнения."""
    positions = np.arange(analysis_slice.start, analysis_slice.stop)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Сравнение: {mut_code} ({mutant_name})", fontsize=16, fontweight="bold")

    axes[0, 0].plot(
        positions, mutant_data[analysis_slice], "r-", linewidth=2,
        marker="o", markersize=5, label=f"Мутант {mut_code}",
    )
    axes[0, 0].plot(
        positions, ref_data[analysis_slice], "b-", linewidth=2,
        marker="s", markersize=5, label="Референс", alpha=0.7,
    )
    if mut_pos is not None and mut_pos - 1 in positions:
        axes[0, 0].axvline(
            x=mut_pos, color="g", linestyle="--", linewidth=1.5,
            alpha=0.7, label=f"Мутация {mut_code}",
        )
    axes[0, 0].set_title(
        f"Профиль агрегации (окно {analysis_slice.start + 1}-{analysis_slice.stop})"
    )
    axes[0, 0].set_xlabel("Позиция аминокислоты")
    axes[0, 0].set_ylabel("Агрегационная склонность")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    bar_colors = ["red" if d > 0 else "blue" for d in difference[analysis_slice]]
    axes[0, 1].bar(positions, difference[analysis_slice], color=bar_colors, edgecolor="black", alpha=0.7)
    axes[0, 1].axhline(y=0, color="black", linewidth=1)
    axes[0, 1].set_title("Разница (Мутант - Референс)")
    axes[0, 1].set_xlabel("Позиция аминокислоты")
    axes[0, 1].set_ylabel("ΔАгрегация")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    if mut_pos is not None:
        mut_idx = mut_pos - 1 - analysis_slice.start
        if 0 <= mut_idx < len(difference[analysis_slice]):
            mut_diff = difference[analysis_slice][mut_idx]
            axes[0, 1].text(
                mut_pos, mut_diff, f"{mut_diff:+.2e}",
                ha="center", fontsize=8, fontweight="bold",
            )

    axes[1, 0].plot(mutant_data, "r-", linewidth=0.5, alpha=0.5, label="Мутант")
    axes[1, 0].plot(ref_data, "b-", linewidth=0.5, alpha=0.5, label="Референс")
    axes[1, 0].axvspan(
        analysis_slice.start, analysis_slice.stop - 1,
        alpha=0.2, color="yellow", label="Окно анализа",
    )
    axes[1, 0].set_title("Полный профиль")
    axes[1, 0].set_xlabel("Позиция аминокислоты")
    axes[1, 0].set_ylabel("Агрегационная склонность")
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compare_with_reference(
    mutant_data: np.ndarray,
    ref_data: np.ndarray,
    mutant_name: str,
    mutation_info: Tuple,
    output_dir: Path,
) -> Tuple[slice, np.ndarray, float]:
    """Сравнивает мутант с референсом и сохраняет график."""
    wt_aa, mut_pos, mut_aa, mut_code = mutation_info

    if len(mutant_data) != len(ref_data):
        min_len = min(len(mutant_data), len(ref_data))
        mutant_data = mutant_data[:min_len]
        ref_data = ref_data[:min_len]
        print(f"  Предупреждение: разная длина, обрезано до {min_len}")

    difference = mutant_data - ref_data

    if mut_pos is not None and mut_pos - 1 < len(difference):
        analysis_slice, peak_pos, peak_val = find_optimal_window(difference, mut_pos - 1)
    else:
        analysis_slice = slice(0, len(difference))
        peak_pos = int(np.argmax(np.abs(difference)))
        peak_val = float(difference[peak_pos])

    fig = create_comparison_plot(
        mutant_data, ref_data, difference,
        analysis_slice, mut_code, mutant_name,
        wt_aa, mut_pos, mut_aa, peak_pos, peak_val,
    )

    safe_name = re.sub(r"[^\w\-.]", "_", mut_code)
    plot_path = output_dir / f"comparison_{safe_name}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return analysis_slice, difference[analysis_slice], peak_val


def run_comparison_analysis(
    input_dir: str,
    reference_file: str,
    output_dir: str = "comparison_results",
    pdf_report: bool = False,
) -> pd.DataFrame:
    """
    Сравнивает все .txt профили в каталоге с референсом.
    Возвращает DataFrame со сводкой.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка референса: {reference_file}")
    ref_data = np.loadtxt(reference_file)
    print(f"  Загружено {len(ref_data)} значений")

    ref_stem = Path(reference_file).stem
    mutant_files = [
        f for f in glob.glob(os.path.join(input_dir, "*.txt"))
        if Path(f).stem != ref_stem
    ]
    print(f"\nНайдено файлов мутантов: {len(mutant_files)}")

    if not mutant_files:
        raise FileNotFoundError(f"Не найдены файлы мутантов в {input_dir}")

    results = []
    pdf_path = output_path / "all_comparisons.pdf"
    pdf_ctx = PdfPages(pdf_path) if pdf_report else None

    try:
        for i, mutant_file in enumerate(mutant_files, 1):
            mutant_name = Path(mutant_file).stem
            print(f"\n[{i}/{len(mutant_files)}] {mutant_name}")

            try:
                mutant_data = np.loadtxt(mutant_file)
                mutation_info = extract_mutation_info(mutant_file)

                analysis_slice, diff_window, peak_diff = compare_with_reference(
                    mutant_data, ref_data, mutant_name, mutation_info, output_path,
                )

                wt_aa, mut_pos, mut_aa, mut_code = mutation_info
                result = {
                    "Файл": mutant_name,
                    "Мутация": mut_code if mut_code else "N/A",
                    "Позиция": mut_pos if mut_pos else "N/A",
                    "Окно_начала": analysis_slice.start + 1,
                    "Окно_конца": analysis_slice.stop,
                    "Размер_окна": analysis_slice.stop - analysis_slice.start,
                    "Макс_дельта": float(np.max(np.abs(diff_window))),
                    "Позиция_макс_дельты": analysis_slice.start + int(np.argmax(np.abs(diff_window))) + 1,
                    "Средняя_дельта": float(np.mean(diff_window)),
                    "Пиковая_дельта": float(peak_diff),
                    "Позиция_пика": int(np.argmax(np.abs(diff_window))) + analysis_slice.start + 1,
                }
                results.append(result)
                print(
                    f"  Окно: {result['Окно_начала']}-{result['Окно_конца']}, "
                    f"|Δ|max = {result['Макс_дельта']:.2e}"
                )

                if pdf_ctx is not None:
                    safe = re.sub(r"[^\w\-.]", "_", mut_code)
                    img_path = output_path / f"comparison_{safe}.png"
                    if img_path.exists():
                        fig = plt.figure(figsize=(14, 10))
                        img = plt.imread(img_path)
                        plt.imshow(img)
                        plt.axis("off")
                        pdf_ctx.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

            except Exception as exc:
                print(f"  Ошибка: {exc}")

    finally:
        if pdf_ctx is not None:
            pdf_ctx.close()
            print(f"\nPDF-отчёт: {pdf_path}")

    if not results:
        raise RuntimeError("Ни один мутант не обработан успешно")

    df = pd.DataFrame(results)
    summary_file = output_path / "summary_results.csv"
    df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"\nСводная таблица: {summary_file}")
    return df


# =============================================================================
# Полный пайплайн
# =============================================================================

def run_full_pipeline(
    fasta_file: str,
    driver_path: Optional[str] = None,
    save_dir: str = "PASTA_Results",
    output_dir: str = "comparison_results",
    reference_file: Optional[str] = None,
    reference_id: Optional[str] = None,
    reference_index: Optional[int] = None,
    headless: bool = True,
    skip_pasta: bool = False,
    tar_path: Optional[str] = None,
    max_sequences: Optional[int] = None,
    wait_timeout: int = DEFAULT_WAIT_TIMEOUT,
) -> dict:
    """PASTA → extract → analyze. Возвращает словарь с путями результатов."""
    fasta_file = str(fasta_file)
    save_dir = str(save_dir)
    output_dir = str(output_dir)

    if not skip_pasta:
        tar_path = submit_all_fasta_at_once(
            fasta_file=fasta_file,
            driver_path=driver_path,
            save_dir=save_dir,
            headless=headless,
            max_sequences=max_sequences,
            wait_timeout=wait_timeout,
        )
        if not tar_path:
            raise RuntimeError("Не удалось получить batch.tar от PASTA")

    if not tar_path or not os.path.exists(tar_path):
        raise FileNotFoundError(f"Архив не найден: {tar_path}")

    profiles_dir = extract_tar_profiles(tar_path)
    ref_path = resolve_reference_file(
        profiles_dir,
        reference_file=reference_file,
        reference_id=reference_id,
        reference_index=reference_index,
        fasta_file=fasta_file,
    )
    print(f"\nРеференс: {ref_path}")

    df = run_comparison_analysis(
        input_dir=str(profiles_dir),
        reference_file=str(ref_path),
        output_dir=output_dir,
    )

    return {
        "tar": tar_path,
        "profiles_dir": str(profiles_dir),
        "reference": str(ref_path),
        "output_dir": output_dir,
        "summary_rows": len(df),
    }



# CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PASTA 2.0 + сравнение профилей агрегации мутантов с референсом",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Полный пайплайн
  python pasta_pipeline.py full mutant.fasta

  # Только PASTA (драйвер подберётся сам; --driver не обязателен)
  python pasta_pipeline.py pasta mutant.fasta

  # Только анализ (уже есть .txt профили)
  python pasta_pipeline.py analyze ./profiles ref.txt -o results

  # Распаковка + анализ из готового tar
  python pasta_pipeline.py extract results.tar --fasta mutant.fasta -o results
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- full ---
    p_full = sub.add_parser("full", help="PASTA → extract → analyze")
    p_full.add_argument("fasta_file", help="Входной FASTA с мутантами")
    p_full.add_argument(
        "--driver",
        default=None,
        help="Путь к msedgedriver.exe (необязательно; иначе Selenium Manager)",
    )
    p_full.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Лимит последовательностей за один запуск PASTA",
    )
    p_full.add_argument(
        "--wait-timeout",
        type=int,
        default=DEFAULT_WAIT_TIMEOUT,
        help=f"Макс. ожидание PASTA в секундах (по умолчанию {DEFAULT_WAIT_TIMEOUT})",
    )
    p_full.add_argument("--save-dir", default="PASTA_Results", help="Каталог для batch.tar")
    p_full.add_argument("-o", "--output", default="comparison_results", help="Каталог графиков/CSV")
    p_full.add_argument("--reference-file", help="Явный .txt референса")
    p_full.add_argument("--reference-id", help="Подстрока в имени референса")
    p_full.add_argument("--reference-index", type=int, default=0, help="Индекс WT в FASTA (0-based)")
    p_full.add_argument("--tar", help="Уже скачанный tar (пропустить PASTA)")
    p_full.add_argument("--no-headless", action="store_true", help="Показать окно браузера")

    # --- pasta ---
    p_pasta = sub.add_parser("pasta", help="Только отправка в PASTA 2.0")
    p_pasta.add_argument("fasta_file")
    p_pasta.add_argument("--driver", default=None, help="Путь к msedgedriver.exe (необязательно)")
    p_pasta.add_argument("--max-sequences", type=int, default=None)
    p_pasta.add_argument("--wait-timeout", type=int, default=DEFAULT_WAIT_TIMEOUT)
    p_pasta.add_argument("--save-dir", default="PASTA_Results")
    p_pasta.add_argument("--no-headless", action="store_true")

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Распаковка tar и опционально анализ")
    p_ext.add_argument("tar_path", help="Путь к batch.tar")
    p_ext.add_argument("--fasta", help="FASTA для автоопределения референса")
    p_ext.add_argument("--reference-file")
    p_ext.add_argument("--reference-id")
    p_ext.add_argument("--reference-index", type=int)
    p_ext.add_argument("-o", "--output", default="comparison_results")
    p_ext.add_argument("--extract-dir", help="Каталог распаковки")
    p_ext.add_argument("--analyze", action="store_true", help="Сразу запустить сравнение")

    # --- analyze ---
    p_an = sub.add_parser("analyze", help="Сравнение .txt профилей с референсом")
    p_an.add_argument("input_dir", help="Каталог с .txt мутантов")
    p_an.add_argument("reference_file", help="Файл референса")
    p_an.add_argument("-o", "--output", default="comparison_results")
    p_an.add_argument("--pdf", action="store_true", help="Собрать PDF со всеми графиками")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "full":
        result = run_full_pipeline(
            fasta_file=args.fasta_file,
            driver_path=args.driver,
            save_dir=args.save_dir,
            output_dir=args.output,
            reference_file=args.reference_file,
            reference_id=args.reference_id,
            reference_index=args.reference_index,
            headless=not args.no_headless,
            skip_pasta=bool(args.tar),
            tar_path=args.tar,
            max_sequences=args.max_sequences,
            wait_timeout=args.wait_timeout,
        )
        print("\n" + "=" * 60)
        print("ПАЙПЛАЙН ЗАВЕРШЁН")
        print("=" * 60)
        for key, val in result.items():
            print(f"  {key}: {val}")

    elif args.command == "pasta":
        tar = submit_all_fasta_at_once(
            fasta_file=args.fasta_file,
            driver_path=args.driver,
            save_dir=args.save_dir,
            headless=not args.no_headless,
            max_sequences=args.max_sequences,
            wait_timeout=args.wait_timeout,
        )
        if tar:
            print(f"\nГотово: {tar}")
        else:
            raise SystemExit(1)

    elif args.command == "extract":
        profiles_dir = extract_tar_profiles(args.tar_path, args.extract_dir)
        if args.analyze:
            ref = resolve_reference_file(
                profiles_dir,
                reference_file=args.reference_file,
                reference_id=args.reference_id,
                reference_index=args.reference_index,
                fasta_file=args.fasta,
            )
            run_comparison_analysis(str(profiles_dir), str(ref), args.output)
        else:
            print(f"Профили: {profiles_dir}")
            print("Для анализа: python pasta_pipeline.py analyze <profiles_dir> <ref.txt>")

    elif args.command == "analyze":
        run_comparison_analysis(
            args.input_dir,
            args.reference_file,
            args.output,
            pdf_report=args.pdf,
        )
        print(f"\nРезультаты: {args.output}")


if __name__ == "__main__":
    main()
