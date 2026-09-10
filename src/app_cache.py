"""
Cache-key helper for the Streamlit app.

The app loads every precomputed artifact from app/data/ through
``@st.cache_data``. Streamlit derives a cache entry's key from
(function qualname + function source + hashes of the call arguments) -- it does
**not** inspect the contents of any file the function reads.

The nightly refresh workflow replaces app/data/*.parquet (and
reported_figures.json) *without changing the loader source*, and Streamlit
Community Cloud reruns the script in the **same process** on a code-only
redeploy. So a cached DataFrame built from the previous data would otherwise
survive until someone hits "Reboot app" -- which is exactly the stale
"+10 runs / 9,395 challenges" render we saw after the last deploy.

``file_signature(path)`` turns a file's (mtime_ns, size) into a small value
that the app passes as an explicit cache-key argument. Replacing the file
(git checkout on redeploy, CI commit) changes mtime_ns, which changes the key,
which forces a fresh read -- no reboot required.

Why (mtime_ns, size) and not a content hash: it is O(1) (a single ``stat``),
and git always writes a fresh modification time when it checks out a changed
file, so it cannot miss a real update. Hashing multi-megabyte parquet on every
rerun would be wasteful and buys nothing for files that only ever change via a
git checkout.
"""
from pathlib import Path


def file_signature(path):
    """Return ``(st_mtime_ns, st_size)`` for ``path`` -- a value that changes
    whenever the file is replaced or modified, suitable as a Streamlit
    cache-key argument.

    Raises ``FileNotFoundError`` if the file is absent; callers that tolerate
    a missing file (e.g. the optional reported_figures.json) must check
    ``path.exists()`` first.
    """
    st = Path(path).stat()
    return (st.st_mtime_ns, st.st_size)
