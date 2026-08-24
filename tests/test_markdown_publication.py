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

    def test_public_research_contract_is_explicitly_conditional(self) -> None:
        """Prevent the public question from drifting back to an unconditional claim."""

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        charter = (PROJECT_ROOT / "reports" / "RESEARCH_CHARTER.md").read_text(
            encoding="utf-8"
        )
        specification = (PROJECT_ROOT / "SPEC.md").read_text(encoding="utf-8")
        literature = (PROJECT_ROOT / "reports" / "LITERATURE_MAP.md").read_text(
            encoding="utf-8"
        )
        plan = (PROJECT_ROOT / "tasks" / "plan.md").read_text(encoding="utf-8")

        self.assertIn("Under what explicit and checkable conditions", readme)
        self.assertIn("## Structured task class", charter)
        self.assertIn("## Conditions to be established", charter)
        self.assertIn(r"\mathfrak K_\ell^*", charter)
        self.assertIn(r"\varepsilon_{\rm cap}", charter)
        self.assertIn("The conclusion is conditional", specification)
        self.assertIn("condition-discovery theorem", plan)
        self.assertIn("The current three categories come from the actual task", charter)
        self.assertIn("task identifiability", charter)
        self.assertIn("finite representability", charter)
        self.assertIn("factor access", charter)
        self.assertIn("Any additional constant must be derived", charter)
        public_contract = f"{readme}\n{charter}\n{specification}"
        self.assertNotIn("five condition", public_contract)
        self.assertNotIn("all five condition", public_contract)
        self.assertNotIn(r"\gamma_*", public_contract)
        self.assertIn("only as one sufficient certificate", plan)
        self.assertIn("does not prove that uniform coercivity", charter)
        self.assertIn("The condition list is an evidence ledger", literature)
        self.assertIn("not final universal assumptions", readme)

        for document in (readme, charter, specification, plan):
            self.assertNotIn("for every task distribution and initialization", document)


if __name__ == "__main__":
    unittest.main()
