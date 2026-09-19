"""HumanEval solution packaging for evalplus.

Evalplus grades sample['solution'] as a complete module (it does not prepend
the prompt). Two packaging bugs made GQH look worse than the generated code:

1. First-fence extraction: ```(?:python)? matches a formula sketch before the
   real ``def`` block (HumanEval/83).
2. Missing prompt imports: GQH often copies ``def f(x: List[...])`` without
   ``from typing import List``, which is a NameError at exec (8/12/20/22).
"""
from __future__ import annotations

import re

try:
    from evalplus.sanitize import sanitize as _evalplus_sanitize
except ImportError:  # tests / environments without evalplus
    _evalplus_sanitize = None


def prompt_imports(prompt: str) -> str:
    lines: list[str] = []
    for ln in prompt.splitlines():
        s = ln.strip()
        if s.startswith("def ") or s.startswith("class "):
            break
        lines.append(ln)
    return "\n".join(lines).rstrip()


def last_def_fence(text: str, entry: str) -> str:
    """Prefer the last fenced block that defines ``entry``; else last fence."""
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    for f in reversed(fences):
        if re.search(rf"^def\s+{re.escape(entry)}\s*\(", f, re.MULTILINE):
            return f
    if fences:
        return fences[-1]
    return text


def extract_completion(raw: str, prompt: str, entry: str) -> tuple[str, str]:
    text = raw
    think = text.rfind("</think>")
    if think >= 0:
        text = text[think + len("</think>"):]
    blob = last_def_fence(text, entry)
    if _evalplus_sanitize is not None:
        san = _evalplus_sanitize(blob, entry) or _evalplus_sanitize(text, entry) or blob.strip()
    else:
        san = blob.strip()
    hdr = prompt_imports(prompt)
    if hdr and hdr not in san and not san.lstrip().startswith("from typing"):
        san = hdr + "\n\n" + san
    if san.startswith(prompt):
        return san, san[len(prompt):]
    return san, san
