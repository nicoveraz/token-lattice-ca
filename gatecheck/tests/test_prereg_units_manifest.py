import json

import numpy as np
import pytest

from gatecheck.prereg import (
    Preregistration, verify_block, evaluate_kills, quarantine, assert_no_smuggling,
)
from gatecheck.units import independence_report, assert_effective_n, unit_level
from gatecheck.manifest import Manifest, Entry, strip_tex_comments
from gatecheck import testing


class TestPrereg:
    def make(self):
        return Preregistration(
            name="dev-transition",
            hypotheses={"H1": "lambda(post) > lambda(pre), Mann-Whitney, run level"},
            frozen={"pre": ["step256", "step512"], "fit_from": 5},
            kill_conditions={"K1": "measure peaks on the degenerate pole"},
            independent_unit="seed",
        )

    def test_block_verifies_and_tamper_detected(self):
        block = self.make().block()
        assert verify_block(block)
        tampered = dict(block, frozen={"pre": ["step1000"], "fit_from": 5})
        assert not verify_block(tampered)

    def test_kills_must_all_be_evaluated(self):
        block = self.make().block()
        with pytest.raises(ValueError, match="never evaluated"):
            evaluate_kills(block, {})
        rep = evaluate_kills(block, {"K1": lambda: False})
        assert rep["survived"] and rep["fired"] == {}
        rep2 = evaluate_kills(block, {"K1": lambda: True})
        assert not rep2["survived"] and "K1" in rep2["fired"]

    def test_quarantine_holds_and_refuses_overwrite(self):
        results = {"H1": 0.9}
        quarantine(results, "peak_as_post", 1.7, "unregistered variant; inflates the effect")
        assert results["_quarantine"]["peak_as_post"]["value"] == 1.7
        with pytest.raises(KeyError):
            quarantine(results, "peak_as_post", 2.0, "again")

    def test_no_smuggling(self):
        block = self.make().block()
        results = {"H1": 0.9, "_provenance": {}, "fit_from": 5}
        assert_no_smuggling(results, block)                     # ok: registered + structural
        results["bonus_effect"] = 3.0
        with pytest.raises(AssertionError, match="bonus_effect"):
            assert_no_smuggling(results, block)
        assert_no_smuggling(results, block, allow={"bonus_effect"})


class TestUnits:
    def test_one_obs_per_unit_is_independent(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=30)
        rep = independence_report(v, np.arange(30), unit_name="seed")
        assert rep.icc == 0.0 and rep.effective_n == pytest.approx(30) and not rep.warn

    def test_grid_from_two_seeds_collapses_to_two(self):
        # the W1 incident: 15 cells from 2 seeds; within-seed values nearly identical
        rng = np.random.default_rng(1)
        seed_effect = {"s21": 0.6, "s22": 0.9}
        values, units = [], []
        for s, mu in seed_effect.items():
            for _cell in range(15):
                values.append(mu + rng.normal(0, 0.01))
                units.append(s)
        rep = independence_report(values, units, unit_name="seed")
        assert rep.n_obs == 30 and rep.n_units == 2
        assert rep.icc > 0.95 and rep.warn
        assert rep.effective_n < 3, "30 correlated cells must not count as 30"
        assert "PSEUDOREPLICATION" in rep.message()
        with pytest.raises(AssertionError):
            assert_effective_n(rep, required=10)

    def test_unit_level_collapse(self):
        labels, means = unit_level([1.0, 3.0, 10.0, 30.0], ["a", "a", "b", "b"])
        assert list(labels) == ["a", "b"]
        assert means == pytest.approx([2.0, 20.0])

    def test_input_validation(self):
        with pytest.raises(ValueError):
            independence_report([1.0], [1])


class TestManifest:
    def test_entry_validation(self):
        with pytest.raises(ValueError):
            Entry(literal="1.58", source="jensen", kind="published")   # no ref
        with pytest.raises(ValueError):
            Entry(literal="x", source="s", kind="guessed")

    def test_strip_tex_comments(self):
        text = "real 0.168 % commented 0.999\nescaped \\% stays 0.5"
        out = strip_tex_comments(text)
        assert "0.999" not in out and "0.168" in out and "0.5" in out

    def test_check_presence_sources_and_recompute(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "dev.json").write_text(
            json.dumps({"plateau": {"lam": 0.16834}, "n_runs": 48})
        )
        m = Manifest()
        m.add("0.168", "results/dev.json", path="plateau.lam", round=3)
        m.add("48", "results/dev.json", path="n_runs")
        m.add("1.580745", "Jensen 1999", kind="published", ref="doi:10/xyz")
        m.add("2x", "ratio of the two lattice sizes", kind="arithmetic",
              note="96/48 stated in prose")

        doc = "we find 0.168 over 48 runs at 2x, vs the published 1.580745"
        rep = m.check(doc, tmp_path, require_all_kinds=True)
        assert rep.ok, rep.message()
        assert rep.counts["measured"] == 2

        # literal absent from the document
        rep2 = m.check("nothing here", tmp_path)
        assert not rep2.ok and "0.168" in rep2.missing_in_document

        # stale manuscript: document says 0.169, source still says 0.168
        m2 = Manifest([Entry(literal="0.169", source="results/dev.json",
                             path="plateau.lam", round=3)])
        rep3 = m2.check("the value is 0.169", tmp_path)
        assert not rep3.ok and rep3.recompute_failures[0]["derived"] == "0.168"

        # missing source file
        m3 = Manifest([Entry(literal="1", source="results/gone.json")])
        rep4 = m3.check("1", tmp_path)
        assert not rep4.ok and rep4.missing_sources == ["results/gone.json"]

    def test_save_load_roundtrip(self, tmp_path):
        m = Manifest()
        m.add("0.5", "results/x.json", path="v", fmt=".1f")
        m.save(tmp_path / "manifest.json")
        m2 = Manifest.load(tmp_path / "manifest.json")
        assert m2.entries[0].literal == "0.5" and m2.entries[0].fmt == ".1f"

    def test_fmt_recompute(self, tmp_path):
        (tmp_path / "r.json").write_text(json.dumps({"gap": 0.6012}))
        m = Manifest([Entry(literal="+0.60", source="r.json", path="gap", fmt="+.2f")])
        assert m.check("gap of +0.60", tmp_path).ok


class TestTestingHelpers:
    def test_retraction_guard(self):
        forbidden = {"capacity axis": ["capacity scales with sensitivity"]}
        testing.assert_retracted_stays_retracted("we claim nothing of the sort", forbidden)
        with pytest.raises(AssertionError, match="capacity axis"):
            testing.assert_retracted_stays_retracted(
                "here capacity scales with sensitivity again", forbidden)

    def test_single_implementation_guard(self, tmp_path):
        a = tmp_path / "canonical.py"
        b = tmp_path / "copycat.py"
        a.write_text("def run_ignited(run):\n    return True\n")
        b.write_text("from canonical import run_ignited\n")
        testing.assert_single_implementation(r"def run_ignited\(", [a, b], allowed_file=a)
        b.write_text("def run_ignited(run):\n    return False\n")
        with pytest.raises(AssertionError, match="more than once"):
            testing.assert_single_implementation(r"def run_ignited\(", [a, b], allowed_file=a)

    def test_assert_fresh_and_manifest(self, tmp_path):
        from gatecheck.results import save_results
        (tmp_path / "analyze.py").write_text("# v1\n")
        out = tmp_path / "results" / "r.json"
        save_results(out, {"v": 0.5}, script=tmp_path / "analyze.py", root=tmp_path)
        testing.assert_fresh(out, tmp_path)
        (tmp_path / "analyze.py").write_text("# v2\n")
        with pytest.raises(AssertionError, match="STALE"):
            testing.assert_fresh(out, tmp_path)

        m = Manifest([Entry(literal="0.5", source="results/r.json", path="v", fmt=".1f")])
        m.save(tmp_path / "manifest.json")
        (tmp_path / "paper.tex").write_text("value 0.5 % and a comment")
        testing.assert_manifest(tmp_path / "manifest.json", tmp_path / "paper.tex", tmp_path)
