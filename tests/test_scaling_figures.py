"""Small rendering contract tests for exported scaling figures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from routing_lab.scaling_figures import render_factorial_effects


class StaticFigureExportTests(unittest.TestCase):
    @staticmethod
    def _effects() -> list[dict[str, object]]:
        """Return a tiny but complete main/interaction rendering fixture."""

        return [
            {
                "term": "width",
                "kind": "main",
                "estimate": 0.10,
                "confidence_interval": [0.05, 0.15],
                "n_pairs": 10,
            },
            {
                "term": "heads:load",
                "kind": "interaction",
                "estimate": -0.08,
                "confidence_interval": [-0.12, -0.04],
                "n_pairs": 10,
            },
        ]

    def test_factorial_figure_exports_real_png_and_searchable_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "factorial"
            paths = render_factorial_effects(self._effects(), stem)
            png = Path(paths["png"])
            svg = Path(paths["svg"])

            self.assertTrue(png.is_file())
            self.assertTrue(svg.is_file())
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            svg_text = svg.read_text(encoding="utf-8")
            self.assertIn("Exploratory normalized embedding-rank contrasts", svg_text)
            self.assertIn("unadjusted pointwise 95% bootstrap CIs", svg_text)

    def test_repeated_render_is_byte_stable(self) -> None:
        """A rerun must not change hashes because of SVG dates or random IDs."""

        with tempfile.TemporaryDirectory() as directory:
            first = render_factorial_effects(self._effects(), Path(directory) / "first")
            second = render_factorial_effects(
                self._effects(), Path(directory) / "second"
            )

            for file_type in ("png", "svg"):
                with self.subTest(file_type=file_type):
                    self.assertEqual(
                        Path(first[file_type]).read_bytes(),
                        Path(second[file_type]).read_bytes(),
                    )


if __name__ == "__main__":
    unittest.main()
