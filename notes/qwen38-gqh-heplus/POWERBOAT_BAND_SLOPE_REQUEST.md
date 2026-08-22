# Request: tok/s + code on the 4.45 bpw band artifact (already on powerboat)

**From Track 1, 2026-08-22.** One run answers two open questions at once, and it needs no new
artifact, no transfer and no code.

## Why this run

1. **Matched-bpw comparison.** Everything so far compared our 3.93 bpw file against
   bartowski IQ4_XS at 4.56 bpw — a 0.63 bpw handicap against us. The artifact below is
   **4.45 bpw**, i.e. within 2.4% of bartowski. It is the fair fight.
2. **The speed/quality slope.** We are choosing where to put a lucebox-tuned build, and on a
   32 GB R9700 with dflash2 the weights budget at 65k ctx is ~24.8 GB — capacity is not the
   constraint, throughput is. We need to know what a step from 3.93 to 4.45 bpw costs in tok/s
   and buys in code before picking a target band.

## The artifact

    powerboat-1:~/gqh-artifacts/qwen38-gqh-32gb-a.gguf
    15,031,638,688 B   (4.45 bpw effective)

Composition: 298x qtype-111 (gqh4) + 10x qtype-108 (gqh3) + **1x qtype-109 (gqh2_h)**, plus
138x q8_0 / 55x q6_k / 3x q5_k, head `output.weight` at gqh3 (521 MB), token_embd q3_k, and the
MTP block present (15 tensors, 265 MB).

**Check before you start: this file needs qtype 109 as well as 108 and 111.** The 13.44 GB
artifacts you have been running carry only 108/111, so if the ROCm loader has 109 unwired it will
refuse this file — that is a one-tensor dependency and worth confirming first rather than
mid-run.

## Important caveat — read this before interpreting the quality number

This artifact was built **2026-08-18**, before all of this week's allocator work (routed-surface
floors, fitted surface weights, the head finding, refit2). Its body allocation is an older
band-exploration plan, not a scaled-up version of our current best.

So: **the tok/s number is representative of the band; the quality number is a LOWER BOUND on
what 4.45 bpw can do.** If code comes out flat or poor, that does not license "the band does not
help" — it licenses "this old allocation at that band does not help". A properly fitted build at
the same band is a 2 h job on our side once we know the speed cost is acceptable.

## What to run

Identical protocol to your last two runs, so the numbers drop straight into the existing table —
please do not change anything else:

* evalplus HumanEval / HumanEval+, 164 tasks, greedy
* SpecLA with the same draft: `qwen38-dflash2-q8_0.gguf`
* GQH I8 default (`GGML_GQH_I8DOT` as you had it), budget capped to `ncols=8`
* **thinking ON, budget 4032** — the primary run, comparable to GQH 149 / IQ4 152
* **thinking OFF** as well if the GPU time is there — completes the 2x2 against GQH 139 / IQ4 146

Report per arm: HumanEval, HumanEval+, AL, tok/s. Plus the failure-overlap counts if they are
cheap (shared fails / GQH-only / IQ4-only) — those have been more informative than the totals.

## Pre-registered reads

* **HE+ >= 152 (thinking on):** the 3.93 bpw band was the whole story — at matched bpw we are at
  parity with IQ4_XS, and the tuned build simply wants this band or a little above.
* **HE+ 150-151:** most of the gap is band, some is allocation/format. Worth one properly fitted
  build at this band before concluding.
* **HE+ ~149 (no change vs 3.93 bpw):** bits are not the lever on code at all, and the deficit is
  format or calibration. That would redirect us to the calibration question (our bundle is
  460/460 thinking-ON traces, code 22%) rather than to bigger artifacts.
* **tok/s:** whatever it is, this is the number that sets the tuned band. If 4.45 bpw still clears
  ~100 tok/s we have real room above it; if it drops toward 90 the tuned build is close to where
  we already are.

## Two optional extras, if there is idle GPU (clearly secondary)

1. **`--pure` IQ4_XS at 13.85 GB, thinking ON.** This is the matched-BYTES arm against our
   13.44 GB file (+3.1%). bartowski's extra 1.72 GB bought 145 -> 146 thinking-off, i.e. nothing,
   so pure-IQ4 thinking-on is probably ~152 — but if it lands ~149 we are at parity at matched
   bytes AND matched regime, which is a materially different public claim from -3.
2. **MTP self-drafting AL vs dflash2 AL**, on any GQH artifact. dflash2 gives AL 6.34-7.46. If
   MTP self-drafting is much worse it is dominated and dropping the MTP block (253-265 MB) from
   the lucebox-tuned variant costs nothing real; if it is comparable, MTP is a single-file
   deployment option worth keeping and that decision flips.

## Exact paths, verified on powerboat 2026-08-22

    target      ~/gqh-artifacts/qwen38-gqh-32gb-a.gguf          15,031,638,688 B
    draft       ~/models/draft/qwen38-dflash2-q8_0.gguf          2,045,471,776 B
    IQ4 arm     ~/models/iq4xs/Qwen3.8-27B-IQ4_XS.gguf          15,567,824,480 B
    (our 3.93)  ~/models/qwen38-gqh-shaped.gguf                 13,440,110,432 B

**Draft-file hazard — there are TWO dflash2 Q8_0 files on the box, 11 MB apart:**

    ~/models/draft/qwen38-dflash2-q8_0.gguf       2,045,471,776 B  <- the one your table used
    ~/models/draft/Qwen3.8-27B-DFlash2-Q8_0.gguf  2,056,414,752 B

Acceptance length depends on the drafter, so switching between these silently would make AL and
tok/s incomparable to the 149/152 pair. Please pin the first one explicitly rather than relying
on a glob or tab-completion.

**Optional extra 1 needs a file that is NOT on the box.** There is no `--pure` IQ4_XS at
13.85 GB; the two IQ4 files present are bartowski at 15.568 GB and a 15.053 GB requant (the
speed-only leftover). Getting the matched-BYTES arm means either pulling #625's published
artifact or running `llama-quantize --pure IQ4_XS` from a bf16 locally. Skip it if that is more
trouble than it is worth — the matched-bpw run above is the one that matters, and I can produce a
`--pure` file on the H100 and ship it over if you would rather not spend the time.
