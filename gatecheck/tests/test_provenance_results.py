import importlib
import json
import pathlib
import sys

import pytest

from gatecheck.provenance import (
    source_sha256, import_closure, environment_fingerprint, stamp, verify_stamp, rel,
)
from gatecheck.results import save_results, load_results, check_no_absolute_paths


@pytest.fixture
def project(tmp_path):
    """A fake project root with one local module and one analysis script."""
    (tmp_path / "mymod.py").write_text("VALUE = 1\n")
    (tmp_path / "analyze.py").write_text("# the analysis\n")
    return tmp_path


class TestProvenance:
    def test_source_sha256_roundtrip(self, project):
        h1 = source_sha256(project / "mymod.py")
        (project / "mymod.py").write_text("VALUE = 2\n")
        h2 = source_sha256(project / "mymod.py")
        assert h1 and h2 and h1 != h2
        assert source_sha256(project / "nope.py") is None

    def test_import_closure_records_actually_imported_local_modules(self, project):
        sys.path.insert(0, str(project))
        try:
            mod = importlib.import_module("mymod")
            importlib.reload(mod)
            closure = import_closure(project)
        finally:
            sys.path.remove(str(project))
            sys.modules.pop("mymod", None)
        assert "mymod.py" in closure
        assert closure["mymod.py"] == source_sha256(project / "mymod.py")
        # nothing outside root leaks in
        assert all(not pathlib.Path(k).is_absolute() for k in closure)

    def test_stamp_and_verify_fresh_then_stale(self, project):
        block = stamp(project / "analyze.py", project)
        assert verify_stamp(block, project).ok
        # the incident this exists for: the source moves on after the run
        (project / "analyze.py").write_text("# edited mid-flight\n")
        report = verify_stamp(block, project)
        assert not report.ok and "analyze.py" in report.stale
        assert "STALE" in report.message()

    def test_verify_reports_missing_files(self, project):
        block = stamp(project / "analyze.py", project)
        (project / "analyze.py").unlink()
        report = verify_stamp(block, project)
        assert not report.ok and "analyze.py" in report.missing

    def test_script_outside_root_is_marked_not_falsely_verified(self, project, tmp_path_factory):
        elsewhere = tmp_path_factory.mktemp("elsewhere") / "runner.py"
        elsewhere.write_text("# lives outside the project\n")
        block = stamp(elsewhere, project)
        assert block["script_external"] is True
        # verification must neither fail on the unverifiable script nor pretend to check it
        assert verify_stamp(block, project).ok

    def test_environment_fingerprint(self):
        env = environment_fingerprint()
        assert env["python"] and "numpy" in env["packages"]

    def test_rel(self, project):
        assert rel(project / "a" / "b.json", project) == "a/b.json"
        outside = pathlib.Path("/somewhere/else.txt")
        assert rel(outside, project) == str(outside)


class TestResults:
    def test_save_then_verify_then_stale(self, project):
        out = project / "results" / "run.json"
        doc = save_results(out, {"lam": 0.168}, script=project / "analyze.py",
                           root=project, independent_unit="seed")
        assert doc["independent_unit"] == "seed"
        loaded, report = load_results(out, root=project, verify=True)
        assert loaded["lam"] == 0.168 and report.ok

        (project / "analyze.py").write_text("# drifted\n")
        _, report2 = load_results(out, root=project, verify=True)
        assert not report2.ok

    def test_unstamped_file_fails_verification(self, project):
        out = project / "results" / "legacy.json"
        out.parent.mkdir()
        out.write_text(json.dumps({"x": 1}))
        _, report = load_results(out, root=project, verify=True)
        assert not report.ok and report.missing

    def test_absolute_path_leak_refused(self, project):
        payload = {"log": f"wrote {project}/results/x.json"}
        with pytest.raises(ValueError, match="leak"):
            save_results(project / "results" / "r.json", payload,
                         script=project / "analyze.py", root=project)
        # and the guard is available standalone
        assert check_no_absolute_paths(payload, [str(project)])
        assert not check_no_absolute_paths({"log": "wrote results/x.json"}, [str(project)])

    def test_prereg_block_embedded(self, project):
        from gatecheck.prereg import Preregistration
        block = Preregistration(name="t", hypotheses={"H1": "up"}).block()
        doc = save_results(project / "results" / "p.json", {"H1": 1.0},
                           script=project / "analyze.py", root=project, prereg=block)
        assert doc["_preregistration"]["name"] == "t"
