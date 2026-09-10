"""
Tests for src/app_cache.file_signature -- the value the Streamlit app threads
into its @st.cache_data keys so a replaced app/data/ file forces a fresh read
(no "Reboot app" needed).

Run: pytest -q
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app_cache import file_signature  # noqa: E402


def test_unchanged_file_has_a_stable_signature(tmp_path):
    p = tmp_path / "artifact.parquet"
    p.write_bytes(b"\x00" * 1024)
    sig1 = file_signature(p)
    sig2 = file_signature(p)
    assert sig1 == sig2
    assert sig1 == (p.stat().st_mtime_ns, 1024)


def test_size_change_changes_the_signature(tmp_path):
    p = tmp_path / "artifact.parquet"
    p.write_bytes(b"\x00" * 1024)
    before = file_signature(p)
    with p.open("ab") as fh:
        fh.write(b"\x00" * 8)          # 1024 -> 1032 bytes
    after = file_signature(p)
    assert after != before
    assert after[1] == before[1] + 8


def test_mtime_change_changes_the_signature(tmp_path):
    # Same content and size, only the modification time moves -- as when git
    # checks out a byte-identical file during a redeploy.
    p = tmp_path / "artifact.parquet"
    p.write_bytes(b"data")
    before = file_signature(p)
    future_ns = before[0] + 5_000_000_000  # +5s, comfortably past fs resolution
    os.utime(p, ns=(future_ns, future_ns))
    after = file_signature(p)
    assert after != before
    assert after[0] == future_ns
    assert after[1] == before[1]          # size unchanged


def test_rewriting_a_file_changes_the_signature(tmp_path):
    # The realistic case: the nightly refresh replaces the file wholesale.
    p = tmp_path / "policy_decomposition.parquet"
    p.write_bytes(b"role_sigma_v1 payload ................")
    before = file_signature(p)
    time.sleep(0.01)
    p.write_bytes(b"role_sigma_v2 payload (different length!) ......")
    after = file_signature(p)
    assert after != before


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_signature(tmp_path / "does_not_exist.parquet")


def test_app_cache_loaders_do_not_underscore_the_signature_param():
    """Streamlit EXCLUDES underscore-prefixed parameters from the @st.cache_data
    key (its documented escape hatch for unhashable handles). If the freshness
    token param in app/streamlit_app.py's _read_parquet / _read_json is ever
    renamed to start with '_', the cache silently goes stale again. Guard that.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent
           / "app" / "streamlit_app.py").read_text()
    tree = ast.parse(src)
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("_read_parquet", "_read_json"):
            arg_names = [a.arg for a in node.args.args]
            seen[node.name] = arg_names
            # exactly (path_str, <freshness token>), token not underscore-prefixed
            assert len(arg_names) == 2, (node.name, arg_names)
            assert not arg_names[1].startswith("_"), (
                f"{node.name}'s freshness-token param {arg_names[1]!r} starts with "
                f"'_', so Streamlit drops it from the cache key -> stale reads")
    assert set(seen) == {"_read_parquet", "_read_json"}, seen
