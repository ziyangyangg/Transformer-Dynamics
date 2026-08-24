"""Publication contracts for readable English Markdown and GitHub math."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
DAMAGED_TEX_ESCAPE = re.compile(r"[\t\r]")


class MarkdownPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (PROJECT_ROOT / "REPOSITORY_SCOPE.toml").open("rb") as handle:
            scope = tomllib.load(handle)
        fixed_paths = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "SPEC.md",
            PROJECT_ROOT / "tasks" / "plan.md",
            PROJECT_ROOT / "tasks" / "todo.md",
            *[PROJECT_ROOT / "reports" / name for name in scope["active_reports"]],
        ]
        result_paths = [
            path
            for directory in scope["active_result_directories"]
            for path in sorted((PROJECT_ROOT / "results" / directory).rglob("*.md"))
        ]
        cls.paths = list(dict.fromkeys([*fixed_paths, *result_paths]))

    def test_publication_documents_exist(self) -> None:
        for path in self.paths:
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertTrue(path.is_file())

    def test_active_documents_are_english_and_free_of_control_bytes(self) -> None:
        for path in self.paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertIsNone(HAN.search(text))
                self.assertIsNone(CONTROL.search(text))
                self.assertIsNone(DAMAGED_TEX_ESCAPE.search(text))

    def test_active_documents_use_github_math_delimiters_only(self) -> None:
        for path in self.paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                for legacy in (r"\(", r"\)", r"\[", r"\]"):
                    self.assertNotIn(legacy, text)

    def test_display_and_inline_math_delimiters_are_balanced(self) -> None:
        for path in self.paths:
            in_fence = False
            in_display = False
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if stripped.startswith(chr(96) * 3):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                if stripped == "$$":
                    in_display = not in_display
                    continue
                with self.subTest(
                    path=path.relative_to(PROJECT_ROOT),
                    line=line_number,
                ):
                    self.assertNotIn("$$", line)
                    if in_display:
                        self.assertNotIn("$", line)
                    else:
                        self.assertEqual(
                            len(UNESCAPED_DOLLAR.findall(line)) % 2,
                            0,
                        )
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertFalse(in_fence)
                self.assertFalse(in_display)


if __name__ == "__main__":
    unittest.main()
