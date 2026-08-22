"""Fail closed when exploratory code silently re-enters the active tree."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from subprocess import check_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RepositoryScopeTests(unittest.TestCase):
    """The public tree must match the reviewed theorem-facing scope exactly."""

    @classmethod
    def setUpClass(cls) -> None:
        with (PROJECT_ROOT / "REPOSITORY_SCOPE.toml").open("rb") as handle:
            cls.scope = tomllib.load(handle)

    @staticmethod
    def _tracked_paths(prefix: str) -> list[str]:
        """Return public paths, excluding private untracked experiment outputs."""

        output = check_output(
            ["git", "ls-files", prefix],
            cwd=PROJECT_ROOT,
            text=True,
        )
        return sorted(line for line in output.splitlines() if line)

    def test_active_python_modules_are_explicit(self) -> None:
        actual = sorted(
            Path(path).name for path in self._tracked_paths("src/routing_lab/*.py")
        )
        self.assertEqual(sorted(self.scope["active_modules"]), actual)

    def test_active_configs_are_explicit(self) -> None:
        actual = sorted(
            str(Path(path).relative_to("configs"))
            for path in self._tracked_paths("configs/*.json")
        )
        self.assertEqual(sorted(self.scope["active_configs"]), actual)

    def test_active_reports_are_explicit(self) -> None:
        actual = sorted(Path(path).name for path in self._tracked_paths("reports/*"))
        self.assertEqual(sorted(self.scope["active_reports"]), actual)

    def test_active_result_directories_are_explicit(self) -> None:
        actual = sorted(
            {Path(path).parts[1] for path in self._tracked_paths("results/*")}
        )
        self.assertEqual(sorted(self.scope["active_result_directories"]), actual)

    def test_temporary_editor_backups_are_absent(self) -> None:
        backups = [path for path in self._tracked_paths("*") if path.endswith(".orig")]
        self.assertEqual([], backups)


if __name__ == "__main__":
    unittest.main()
