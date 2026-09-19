# Request: establish the same-arm floor on HumanEval+ (repeat one arm)

**From Track 1, 2026-08-22.** Highest-value next run, and it is a repeat rather than anything new.

## Why: the -3 we have all been explaining has no error bar

Every HumanEval+ number so far is **n=1**:

    3.93 GQH think-on   149
    4.45 GQH think-on   149
    IQ4_XS  think-on    152

No arm has been repeated, so the same-arm floor on this instrument is **unknown**. On our AIME
gate that floor is **+-3 items**, established by repeating both a vendor arm and a GQH arm. If
HumanEval+ behaves similarly, the -3 gap is *inside* the floor and there may be no effect to
explain at all — and we have already spent three measurements and two hypotheses on it.

Two hints that variance here is nonzero: 3.93 I8 vs f32 think-off differ by 1 item on the SAME
weights (arithmetic the only difference), and greedy is not bit-deterministic under continuous
batching.

## What to run

**Repeat the 3.93 GQH think-on arm** — the one every current claim rests on. Byte-identical
artifact, byte-identical config, nothing changed:

    model   ~/models/qwen38-gqh-shaped.gguf               13,440,110,432 B
    draft   ~/models/draft/qwen38-dflash2-q8_0.gguf        2,045,471,776 B   <- pin explicitly
    evalplus HumanEval / HumanEval+, 164 tasks, greedy, SpecLA
    GQH I8 default, budget capped ncols=8, thinking ON, budget 4032

**Pin the draft path explicitly** — there are two dflash2 Q8_0 files 11 MB apart and the other one
(`Qwen3.8-27B-DFlash2-Q8_0.gguf`, 2,056,414,752 B) would change AL and silently break
comparability with the 149.

Two runs of that arm gives a spread; three gives a usable floor. If GPU time allows, **also repeat
IQ4_XS think-on** — on AIME the floor turned out similar for vendor and GQH arms, but that was
measured, not assumed, and the gap we care about is between the two arms.

## Please capture PER-ITEM pass/fail, not just totals

This is the part that matters most, and it is cheap if evalplus is already writing per-task
results. Comparing two repeats item-by-item separates two very different worlds:

* **Same items fail both times** -> the instrument is tight, the arm is deterministic in practice,
  and a -3 total difference is meaningful.
* **Different items fail each time** -> the totals are drifting over a churning set, and n=1
  differences of a few items are noise regardless of what the totals look like.

The totals alone cannot distinguish those. On AIME the item-level view is what told us capping,
not correctness, was driving early comparisons.

## Pre-registered reads

* **Spread 0-1 items:** instrument is tight. The -3 vs IQ4 is real, and the format/calibration
  question is live — that is when a calibration experiment becomes worth its GPU cost.
* **Spread 2-3 items:** -3 is at or inside the floor. Nothing to explain yet; any code claim
  (ours or about the gap) needs n=3 per arm before it means anything.
* **Spread >3 items:** the instrument is too noisy at n=1 for ANY of the comparisons we have
  drawn from it, including the band-slope conclusion that bits do not help on think-on. That
  would need re-examining at n=3.

## Note on what this retroactively affects

If the floor is +-3, then "IQ4_XS is +3 on HumanEval+" was never established, and neither was any
code deficit on our side. Our HF card currently makes no code claim at all, which turns out to
have been the right call — but the internal narrative has been treating -3 as a fact, and this run
is what decides whether it is one.
