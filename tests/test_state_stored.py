"""Load-bearing results files must carry the object their numbers were reduced from.

WHY THIS IS A REGISTRY AND NOT A GREP. The obvious gate is a grep over `experiments/` for
`run(...)["final"]` consumed only through a scalar reducer. Fifty-five scripts match. Most of them
produced results that are already published, and re-running them costs real compute for questions
nobody is asking -- so a grep-gate would go red on day one and be silenced with a suppression
comment in every file, which is a gate that trains people to ignore it.

The defect is not "a script reduced a state". Reducing is the point of a readout. The defect is
"a LOAD-BEARING result cannot be re-questioned without a re-run", and load-bearing is a property of
the result, not of the code. So the gate is a list, it is short, and adding to it is a decision
someone makes on purpose.

THE THREE INSTANCES THIS DESCENDS FROM (F116, the remote share campaign, F136), all one mechanism:
the largest object the measurement produces is discarded, so its defects are invisible and every
new question costs a full re-run. The second of those shipped a whole campaign reading orbit
lengths without knowing it.
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gatecheck" / "src"))
from gatecheck import has_state, unpack_state, STATE_KEY          # noqa: E402
sys.path.pop(0)

# THE REGISTRY. A results file earns a place here when a finding rests on it AND a plausible future
# question is about the state rather than the scalars. Keep the reason: it is what a later reader
# needs in order to judge whether the entry still applies.
LOAD_BEARING = {
    "share_invariance.json": (
        "F130's grid -- the instrument's model-attributable readout. F136 asked whether top1 is "
        "1/period on a crystallised lattice and this file could not answer, which cost a re-run."),
}

# DELIBERATELY NOT IN THE REGISTRY, with the reason, because an unexplained absence is the same
# defect as an unexplained entry:
#
#   topk_ablation.json -- F134's 320-cell grid. Its conclusion is about RANKINGS, and the period
#     question was answered for it by a cheap screen over stored scalars that excluded 305 cells
#     outright (pooling over replicas means a 1/period reading needs a small distinct count). Only
#     the 15 the screen could not exclude were re-run with state. Requiring all 320 would mandate a
#     four-hour backfill to answer a question already answered, which is how a gate stops being
#     worth obeying. The script stores state from now on, so the file converges on its own.


def cells(name):
    p = ROOT / "results" / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    return json.load(open(p)).get("cells", {})


@pytest.mark.parametrize("name,reason", sorted(LOAD_BEARING.items()))
def test_every_cell_carries_its_state(name, reason):
    missing = [k for k, c in cells(name).items() if not has_state(c)]
    assert not missing, (
        f"{name} has {len(missing)} cell(s) with no stored state, e.g. {missing[:3]}. {reason} "
        f"Re-run the producing script: it treats a state-less cell as incomplete and refills it.")


@pytest.mark.parametrize("name,reason", sorted(LOAD_BEARING.items()))
def test_stored_state_is_readable_and_shaped(name, reason):
    for k, c in list(cells(name).items())[:8]:
        arr = unpack_state(c[STATE_KEY])
        assert arr.ndim == 2 and arr.size > 0, f"{name}:{k} stored an unusable state block"
        assert list(arr.shape) == c[STATE_KEY]["kept_shape"]


def test_stored_state_reproduces_the_scalars_it_was_reduced_to():
    """The state must be the state THESE numbers came from, not a state from some other run.

    This is the check that makes the convention worth anything: a stored array that does not
    reproduce the cell's own reported scalars is worse than no array, because it invites
    re-analysis of the wrong object.
    """
    import numpy as np
    bad = []
    for k, c in cells("share_invariance.json").items():
        if not has_state(c) or not c[STATE_KEY]["complete"]:
            continue                                   # a strided state cannot reproduce a pooled count
        a = unpack_state(c[STATE_KEY])
        _, cnt = np.unique(a.reshape(-1), return_counts=True)
        top1 = float(cnt.max() / cnt.sum())
        rep2 = float(np.mean(a[:, :-1] == a[:, 1:]))
        if abs(top1 - c["top1"]) > 1e-12 or abs(rep2 - c["rep2"]) > 1e-12:
            bad.append((k, c["top1"], top1, c["rep2"], rep2))
    assert not bad, f"stored state does not reproduce its own scalars: {bad[:3]}"


def test_the_registry_names_its_reasons():
    """An entry with no reason is a rule nobody can later judge, so it is not allowed to be one."""
    for name, reason in LOAD_BEARING.items():
        assert len(reason) > 40, f"{name} needs a reason a later reader can act on"
