# Qwen3.8-27B GQH HumanEval+ notes (powerboat, 2026-08-22)

Staging branch `feat/gqh-qwen38-dspark`: GQH qtypes 108–111 + Luce-Org/lucebox#625
(DFlash2/SpecLA), I8 N=8 default, SpecLA budget capped so verify stays ncols=8.

These notes are **not** a merge claim. They pin the eval so a later pass can
fold this branch into main without re-deriving the −3 vs IQ4.

## Pins

| | path | bytes |
|---|---|---|
| GQH 3.93 | `~/models/qwen38-gqh-shaped.gguf` | 13,440,110,432 |
| GQH 4.45 | `~/gqh-artifacts/qwen38-gqh-32gb-a.gguf` | 15,031,638,688 |
| IQ4_XS | `~/models/iq4xs/Qwen3.8-27B-IQ4_XS.gguf` | 15,567,824,480 |
| draft | `~/models/draft/qwen38-dflash2-q8_0.gguf` | **2,045,471,776** (not the 2,056,414,752 sibling) |

Protocol: greedy, thinking ON budget 4032, max_out 8192, SpecLA `--ddtree-budget 8`
(capped to 7 nodes), I8 default, same binary `dflash_server`.

## Same-arm floor (think-on)

Greedy sequential SpecLA is bit-deterministic here. Spread **0**.

| arm | n | HE | HE+ | AL | tok/s |
|---|---|---|---|---|---|
| 3.93 GQH | 3 | 156 | 149 | 6.360 | ~120 |
| IQ4_XS | 2 | 161 | 152 | 6.343 | ~112 |

r1/r2/r3 GQH jsonl is byte-identical (0/164 SHA diffs). AIME’s ±3 does **not**
apply. See `FLOOR_REPORT.json`.

## The −3 was packaging, not bits

Evalplus uses `sample['solution']` as a complete module. The first-fence
extractor + missing prompt `from typing` imports turned four correct GQH
functions into `NameError` and scored HumanEval/83’s formula sketch instead of
the last `def` fence.

Same 3.93 generations, `harness/he_extract.py`:

| packaging | GQH HE+ | IQ4 HE+ |
|---|---|---|
| published (first fence, no imports) | 149 | 152 |
| + prompt typing header | 153 | 152 |
| + last fence that defines the entry | **154** | 152 |

4.45 GQH think-on stays 149 published / 150 with import header (HE/20 was the
same missing-import miss; 8/12/22/83/163 already passed).

Full item write-up: `HUMANEVAL_SIX_FAILS.md`. After packaging, GQH-only leftover
is **HumanEval/163** (range scan vs digit scan). IQ4-only: 55, 134, 151.

## Speed (same protocol, I8 on)

4.45 is GQH4-heavy (`pair-fused gate/up gqh4`); 3.93 fuses **gqh3**. Think-on
112 tok/s on 4.45 is the fatter band + AL 6.36, not an f32 fallback (3.93 f32
think-off was 105.5). Think-off 4.45 I8 is 126 vs 3.93 I8 137 vs f32 105.

## Files here

- `FLOOR_REPORT.json` — r1/r2/r3 item fails, SHA diffs, IQ4 r2
- `SIX_FAILS_REPORT.json` — the six named GQH-only tasks
- `HUMANEVAL_SIX_FAILS.md` — prose for geo-quant
- `IMPORT_HEADER_RESCORE.json` — 149→153 / IQ4 unchanged
- `six_full_replies.json` — full content+reasoning for the six
- `samples/gqh_r2.jsonl`, `iq4_r2.jsonl` — published packaging
- `samples/gqh_r2_reextract6.jsonl` — 154 packaging
