"""
checkpoint.py - resumable computations
======================================
Every long computation of the repository saves its work as it goes, so that a run which stops
(interruption, crash, power cut) continues where it stopped when it is launched again:
  * files are written atomically (to a temporary file in the same folder, then renamed), so a stopped
    run never leaves a truncated file behind;
  * a stage folder keeps the settings and the code it was computed with (SETTINGS.json); resuming with
    other settings or another code is refused instead of mixing two computations;
  * a complete stage writes a MANIFEST.json, whose fingerprint identifies its results downstream;
  * a result file of the analyses is reused only if it was computed from the same inputs (a side file
    <name>.source holds their fingerprint).

Author: Olivier Vertu Ndinga Oba
"""
import ast
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


_UMASK = os.umask(0)
os.umask(_UMASK)


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.chmod(tmp, 0o666 & ~_UMASK)                 # the permissions of an ordinary file, not mkstemp's 0600
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_pickle(obj, path):
    """pickle.dump, atomically."""
    _atomic(path, lambda fh: pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL))


def save_text(text, path):
    _atomic(path, lambda fh: fh.write(text.encode()))


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def file_fingerprint(path):
    """sha1 of a file's content, cached in <path>.sha1 against its size and modification time."""
    path = Path(path)
    st = path.stat()
    key = f"{st.st_size} {st.st_mtime_ns}"
    cache = path.with_name(path.name + ".sha1")
    if cache.exists():
        k, _, digest = cache.read_text().strip().rpartition(" ")
        if k == key:
            return digest
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    save_text(f"{key} {h.hexdigest()}", cache)
    return h.hexdigest()


def code_fingerprint(*files):
    """sha1 of the executable code of Python files: comments, docstrings and layout are ignored, so a
    change of wording does not invalidate results, while any change of the computation does."""
    h = hashlib.sha1()
    for f in files:
        tree = ast.parse(Path(f).read_text())
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                    and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        h.update(ast.dump(tree, annotate_fields=False, include_attributes=False).encode())
    return h.hexdigest()


def _canonical(settings):
    return json.dumps(settings, sort_keys=True, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def guard_settings(folder, settings, pattern="*.pkl"):
    """Record the settings of a stage folder, or check that they are those of the files already there.
    A folder that holds results but no SETTINGS.json (unknown provenance) is refused as well."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SETTINGS.json"
    text = _canonical(settings)
    if not path.exists() and any(folder.glob(pattern)):
        raise RuntimeError(f"{folder} holds results of unknown settings (no SETTINGS.json). "
                           f"Move that folder away (or delete it) to compute again with the current ones.")
    if path.exists():
        if path.read_text() != text:
            old = json.loads(path.read_text())
            new = json.loads(text)
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise RuntimeError(f"{folder} holds results computed with other settings or code ({', '.join(diff)}). "
                               f"Move that folder away (or delete it) to compute again with the current ones.")
    else:
        save_text(text, path)


def write_manifest(folder, files):
    """Mark a stage folder complete: its settings and the list of its files with their sizes."""
    folder = Path(folder)
    entries = sorted((Path(f).name, Path(f).stat().st_size) for f in files)
    body = dict(settings=json.loads((folder / "SETTINGS.json").read_text()), files=entries)
    save_text(_canonical(body), folder / "MANIFEST.json")


def stage_fingerprint(folder):
    """Fingerprint of a complete stage folder (its MANIFEST.json)."""
    path = Path(folder) / "MANIFEST.json"
    if not path.exists():
        raise RuntimeError(f"{folder} is incomplete (no MANIFEST.json): run that stage to the end first.")
    return hashlib.sha1(path.read_bytes()).hexdigest()


class Store:
    """A dict of results saved to `path` at every update, reused only for the same inputs (`source`)."""

    def __init__(self, path, source):
        self.path = Path(path)
        self.source = str(source)
        self._src = self.path.with_name(self.path.name + ".source")
        self.data = {}
        if self.path.exists() and self._src.exists() and self._src.read_text().strip() == self.source:
            self.data = load_pickle(self.path)

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        self.save()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def save(self):
        save_pickle(self.data, self.path)            # the data first, then the fingerprint it belongs to
        save_text(self.source, self._src)
