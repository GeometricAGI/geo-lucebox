# Route back to upstream — what we carry, what can go home, what never will

**2026-08-24.** `geo-lucebox` is not a long-lived fork; it is our development staging of
`Luce-Org/lucebox`. The target state is: **our work lands upstream, `main` returns to tracking
`upstream/main`, and local topic branches are deleted.** This file records how far we are from
that and the order of operations, because the branch topology had become hard to read.

## Where we actually are

    upstream/main   53d12032
    origin/main     bba7479c   20 commits behind upstream, 39 commits ahead of the merge base

**None of our 36 non-merge commits exist upstream** (`git cherry upstream/main origin/main` →
36 `+`, 0 `-`). So nothing can be dropped as already-landed; every one is still ours to place.

## The single fact that de-confuses this

Merging the GQH staging branch into `main` conflicts in 6 files. Merging **plain upstream** into
`main` conflicts in 5. **Four are the same files** — `server/CMakeLists.txt`,
`chat_template.cpp`, `http_server.cpp`, `tool_parser.cpp` (+ `test_server_unit.cpp`). Only two are
genuine GQH reconciliation (`model_capabilities.h`, `draft_graph.cpp`).

And those four conflict **because of muse-glimmer**: each of `chat_template.cpp` and
`tool_parser.cpp` has exactly one commit from us and it is the muse/ATEM one; `http_server.cpp`
has three, one of which is muse. Upstream meanwhile rewrote the same files heavily
(`http_server.cpp` +452/−77, `test_server_unit.cpp` +581/−9).

**So the conflict is not GQH and not upstream drift in general. It is the muse-glimmer server
integration sitting on `main` where upstream refactored underneath it.** Deciding muse's fate is
the one move that unblocks everything else.

## What we carry, tiered by where it can go

| tier | commits | what | destination |
|---|---|---|---|
| general fixes | 4 | tokenizer digit-grouping; tool-call turn replay; DSpark AR fallback visibility | **upstream now** — small, model-agnostic |
| muse-glimmer backend | 27 | GGUF loader, forward graphs, KV cache, ATEM chat format + tool dialect, DFlash spec decode, docs | upstream as a feature PR, or accept as permanently ours |
| custom qtypes | 5 | dmix2 sidecar, mix-qtype matvec/MMQ perf | **never upstream** — tied to qtypes 105/106 |
| GQH (other branches) | ~48 | qtypes 108–111, fused matvec, header KV | **never upstream** — our private format |

The last two tiers matter for the goal: **`main` can never fully return to `upstream/main` while
also serving our artifacts**, because qtypes 105/106 and 108–111 are ours. The reachable target is
`main` = upstream mirror, with the custom-qtype layer on a long-lived branch that rebases on
upstream periodically — not merged into main.

## DECISION 2026-08-24: muse-glimmer stays ours for now

Upstream has not the bandwidth to take a new model backend yet, but it may be useful to them
later. So muse is **kept, not offered** — and the job becomes keeping it *offerable* rather than
extracting it now.

**Two findings that change the plan recorded below.**

**1. muse and the custom-qtype layer are ONE workstream, not two.** The `feat/muse-glimmer-serving`
branch was merged and deleted, so the work exists only as a range on `main` — and it is
interleaved with the mix-qtype commits, not contiguous. Several commits are both at once:
`mix MMQ: scope the enable to the muse backend`, `test(mix): add the missing MMQ correctness gate;
measure muse best-serving config`. The mix-MMQ work exists partly *because* muse needed it. So the
offerable unit is "muse-glimmer serving **plus** the mix-qtype support it depends on", and the
qtype half cannot go upstream (105/106 are ours). **An upstream offer therefore needs a decoupling
decision first — whether muse can serve stock qtypes with the mix path as an optional
optimisation — and that is design work, not extraction work.** Doing the extraction now would be
speculative effort for an offer that may never be made, and would likely produce a branch that
does not build.

**2. So `main` must NOT be reset to upstream.** The earlier version of this file recommended
that; it was wrong. `main` is the integration point we serve from (muse + custom qtypes), so the
direction of travel is the opposite: **`main` absorbs upstream periodically**, and the four
server-layer conflicts get resolved once per sync rather than designed away.

Current state is tagged `integration/main-pre-upstream-sync-20260824` (= `bba7479c`) so this
integration is recoverable by name before any sync attempt.

**What the resolution actually involves** — gauged on the smallest conflict rather than guessed.
In `tool_parser.cpp` we add an ATEM opener (`<atem:function_calls>`) with a hardcoded
longest-opener length; upstream refactored the same detection into named constants
(`FUNCTION_CALL_OPEN`, `FUNCTION_CALLS_OPEN`, `BARE_FUNCTION_OPEN`) with a compare-based check. So
resolution is **"keep both, expressed in upstream's structure"** — add ATEM as another constant
and re-derive the length logic. Tractable, but it is a small rewrite per file rather than a
pick-a-side, and `test_server_unit.cpp` is the awkward one (upstream +581/-9).

**Verification gap to respect:** these conflicts sit in the tool-call streaming path, where a
wrong resolution breaks serving subtly rather than loudly. This box has cmake but no configured
build for this project, so a resolution done here lands **unverified**. It needs a build plus the
server unit tests on a box with the toolchain (powerboat or a lucebox) before it is merged.

## Order of operations

1. **Done: two clean upstream candidates exist.** Both cherry-pick onto current `upstream/main`
   with zero conflicts and are pushed as single-commit branches:
   * `for-upstream/tokenizer-digit-grouping` (`server/src/server/tokenizer.{cpp,h}`)
   * `for-upstream/preserve-tool-call-turns` (`server/src/server/http_server.cpp`)
   These are ready to open against `Luce-Org/lucebox` and are the cheapest divergence reduction
   available.
2. **Muse-glimmer: decided — ours for now** (see above). It stays on `main`; the consequence is
   that each upstream sync re-resolves the same four server files until upstream's refactor and
   our ATEM integration are reconciled once and carried forward.
3. **Sync `main` with `upstream/main`** (20 commits behind), resolving the four server-layer
   files in upstream's structure as described above, then build and run the server unit tests
   before merging. Once `main` carries current upstream, the GQH staging branch (PR #34) rebases
   onto it and its conflict surface should reduce to the two genuine GQH files.
4. Retire the stacked GQH perf chain into the staging branch: PR #32
   (`perf/gqh-matvec-gfx1201` → `feat/gqh-qtypes`) and #33 (`perf/gqh-multicol-port` → #32) are
   not in `feat/gqh-qwen38-dspark`, so "our current implementation" is currently split across
   three branches.

## Notes for whoever picks this up

* Nothing on powerboat is unpushed — `~/geo-lucebox`, `~/lb-graphs` and `~/lb-upstream` are three
  **worktrees** of one clone, at or slightly behind origin. `~/lb-upstream` is
  `feat/gqh-qwen38-dspark`, i.e. PR #34's branch.
* `~/projects/lucebox-hub` on this box has uncommitted work on
  `feat/ds4-adaptive-on-upstream` — left untouched.
* PR #34 reports `mergeable: false / dirty` for the reason in the section above, not because the
  GQH work is broken.
