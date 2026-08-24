# The 6 GQH-only HumanEval+ fails (3.93 think-on vs IQ4)

Same-arm floor was 0: 149 vs 152 is sticky. These 6 items were the whole named gap.

**5/6 were eval packaging, not wrong code.** Same 3.93 generations, honest packaging:

| | HE | HE+ |
|---|---|---|
| published (first fence, no prompt imports) | 156 | **149** |
| + prompt `from typing` header | 160 | **153** |
| + last fence that defines the entry | 161 | **154** |
| IQ4 (unchanged) | 161 | **152** |

## Per item

| task | kind | what happened |
|---|---|---|
| **8** | harness | Correct `sum`/`product`. GQH omitted `from typing import List, Tuple`. Evalplus `NameError` at exec (empty fail_tests). IQ4 and 4.45 copied the imports. |
| **12** | harness | Correct `max(..., key=len)`. Missing `Optional`/`List`. |
| **20** | harness | Correct sorted adjacent-min. Missing `List`/`Tuple`. 4.45 had the same miss (that is why 20 was in both GQH bands; header merge takes 4.45 149→150). |
| **22** | harness | Correct `isinstance(x, int) and not isinstance(x, bool)`. Missing `List`/`Any`. |
| **83** | harness | Model wrote the right function as the **last** fenced block (`return 18 * 10 ** (n-2)`). Extractor took the **first** fence, a unicode formula sketch. 4.45 already emitted a `def` first. |
| **163** | **model** | Understood “even digits” as `{0,2,4,6,8}` but implemented `for num in range(low, high+1)`, which times out on plus input `[987654321, 123456789]`. Canonical / IQ4 / 4.45 iterate `range(0, 10)`. **Only remaining 3.93-only HE+ miss.** |

## Overlap after packaging (3.93 vs IQ4)

- shared fails (9): 10, 32, 39, 91, 97, 132, 141, 145, 154
- GQH-only: **163**
- IQ4-only: 55, 134, 151

The −3 was not a format/calibration deficit on code. One real 3.93 miss remains, and IQ4 has three of its own.

Dumps in this directory: `six_full_replies.json`, `SIX_FAILS_REPORT.json`, `samples/gqh_r2_reextract6.jsonl`.
