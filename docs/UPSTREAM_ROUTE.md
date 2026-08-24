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

## Order of operations

1. **Done: two clean upstream candidates exist.** Both cherry-pick onto current `upstream/main`
   with zero conflicts and are pushed as single-commit branches:
   * `for-upstream/tokenizer-digit-grouping` (`server/src/server/tokenizer.{cpp,h}`)
   * `for-upstream/preserve-tool-call-turns` (`server/src/server/http_server.cpp`)
   These are ready to open against `Luce-Org/lucebox` and are the cheapest divergence reduction
   available.
2. **Decide muse-glimmer.** Upstream feature PR, or accept it as ours and move it off `main` onto
   a long-lived branch. Until this is decided, every `main` merge re-fights the same four files.
3. **Then** `main` can be reset to track `upstream/main`, and the GQH staging branch (PR #34,
   `feat/gqh-qwen38-dspark`) rebases onto it — at which point its conflict surface should reduce
   to the two genuine files.
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
