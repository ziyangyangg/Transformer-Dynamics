"""Publication receipts for the externally verified single-layer counterexample."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SingleLayerProofBundleTests(unittest.TestCase):
    def test_verified_proof_hash_and_external_verdict_are_frozen(self) -> None:
        proof = ROOT / "proofs" / "MATRIX_MQAR_SMALL_INIT_COUNTEREXAMPLE_PROOF.md"
        self.assertEqual(
            hashlib.sha256(proof.read_bytes()).hexdigest(),
            "0a7029fdd72527309efbc03d70e0eac8106490cc55fb15b83901aa403d97cc8b",
        )
        receipt = (
            ROOT
            / "verification"
            / "matrix_mqar_small_initialization"
            / "verification_reports.jsonl"
        )
        records = [json.loads(line) for line in receipt.read_text().splitlines()]
        final_record = records[-1]["record"]
        report = final_record["verification_report"]
        self.assertEqual(final_record["verdict"], "correct")
        self.assertEqual(report["critical_errors"], [])
        self.assertEqual(report["gaps"], [])

    def test_independent_population_audit_passes(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "n0_small_initialization_audit.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["episode_count"], 48)
        self.assertLess(report["maximum_discrepancy"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()
