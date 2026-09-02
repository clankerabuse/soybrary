"""Point every test at a throwaway data directory before the app is imported."""

import atexit
import os
import shutil
import tempfile

_TMP_DATA_DIR = os.environ.get("SOYBRARY_DATA_DIR")
if not _TMP_DATA_DIR:
    _TMP_DATA_DIR = tempfile.mkdtemp(prefix="soybrary-test-")
    os.environ["SOYBRARY_DATA_DIR"] = _TMP_DATA_DIR
    atexit.register(shutil.rmtree, _TMP_DATA_DIR, ignore_errors=True)
