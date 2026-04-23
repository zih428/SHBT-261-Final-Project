from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

# Compatibility shim for repo-local `python -m textvqa_proj.cli ...` calls.
# The canonical source package is `src/textvqa_proj`; this top-level package
# only exposes that source tree when the project has not been installed.
__path__ = extend_path(__path__, __name__)
_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "textvqa_proj"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

__version__ = "0.1.0"
