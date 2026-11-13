"""
progress.py - progress display
==============================
Author: Olivier Vertu Ndinga Oba
"""

def progress(label, current, total):
    percent = 100.0 * current / total if total else 100.0
    print(f'    {label}: {current}/{total} ({percent:5.1f}%)', flush=True)