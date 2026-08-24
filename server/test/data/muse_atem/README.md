# ATEM reference prompts (Muse-Glimmer)

Ground-truth prompt renderings produced by the **model's own** chat template,
consumed by `test/test_muse_atem_reference_prompts.cpp` for a byte-exact
comparison against lucebox's native ATEM renderer.

`cases.json` records the exact message/tool inputs for each `<name>.txt`.

Regenerate (needs the HF snapshot and `transformers`):

```bash
python - <<'EOF'
import json, pathlib
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B")
d = pathlib.Path("server/test/data/muse_atem")
for name, c in json.loads((d / "cases.json").read_text()).items():
    (d / f"{name}.txt").write_text(tok.apply_chat_template(
        c["messages"], tools=c["tools"], add_generation_prompt=True,
        tokenize=False, current_date="2026-08-12"))
EOF
```

`current_date` is pinned because the synthesized system turn carries a
dateline; the renderer reads `DFLASH_ATEM_CURRENT_DATE` so the test can match
it (CMake sets it, and the same override makes eval harnesses reproducible).

These fixtures caught three real renderer bugs on first run: nlohmann's
`dump()` sorts object keys and omits separator spaces (the reference
preserves insertion order with `", "` / `": "`), assistant turns are always
addressed `to=user` rather than bare, and a tool-call turn ends with `<|eot|>`
when a `tool` turn follows — not the `<|eom|>` a reading of the template's
`end_token` alone suggests.
