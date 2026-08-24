from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from he_extract import extract_completion, last_def_fence, prompt_imports  # noqa: E402


class HeExtractTest(unittest.TestCase):
    def test_prompt_imports_stops_at_def(self) -> None:
        prompt = "from typing import List, Tuple\n\ndef foo(x: List[int]) -> Tuple[int, int]:\n    pass\n"
        self.assertEqual(prompt_imports(prompt), "from typing import List, Tuple")

    def test_last_def_fence_skips_formula_sketch(self) -> None:
        raw = (
            "```\n10^(n-1) + 9·10^(n-2)\n```\n\n"
            "```python\ndef starts_one_ends(n):\n    if n == 1:\n        return 1\n"
            "    return 18 * 10 ** (n - 2)\n```\n"
        )
        blob = last_def_fence(raw, "starts_one_ends")
        self.assertIn("def starts_one_ends", blob)
        self.assertNotIn("10^(n-1)", blob)

    def test_extract_prepends_typing_imports(self) -> None:
        prompt = "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
        raw = (
            "```python\n"
            "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
            "    return (sum(numbers), 1)\n"
            "```\n"
        )
        sol, _ = extract_completion(raw, prompt, "sum_product")
        self.assertTrue(sol.lstrip().startswith("from typing import List, Tuple"))
        self.assertIn("def sum_product", sol)


if __name__ == "__main__":
    unittest.main()
