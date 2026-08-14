"""Is a readout measuring the MODEL, or the construction it is measured in?

THE QUESTION THIS ANSWERS. Anyone who runs a language model in a self-feeding loop -- a ring, a
self-consistency vote, an iterated-refinement chain, a multi-turn agent -- can compute statistics of
the loop's behaviour and report them as properties of the model. Most such statistics are properties
of the LOOP. They move when the construction changes and barely move when the model does, and
nothing in a correct calculation objects.

The project this package was extracted from measured that split on two of its own readouts and got
opposite answers. A Lyapunov-style exponent had a cross-model spread of 0.051 against a
construction-induced range of 0.68 -- a ~30x ratio -- with a seed-to-seed ranking stability of 0.030,
i.e. a real spread carrying no reproducible ordering. An attractor-share readout on the same models
and the same constructions had signal on 6 of 6 constructions, ranking stability 0.848, and
cross-construction agreement +0.752. Same lattice, same models, same seeds: one readout is about the
apparatus and the other is about the model, and only this test separates them.

THE TWO AXES.
  vary the CONSTRUCTION, fix the model  -> how much of the readout is apparatus
  vary the MODEL, fix the construction  -> how much of it is the thing you meant to measure

THE ORDER IS NOT NEGOTIABLE, and it is the part most easily got wrong. A first version of the
project's own analysis asked for cross-construction INVARIANCE directly, which collapsed the case
that actually occurred: a spread that exceeds noise while producing no reproducible ordering. A
spread you cannot rank is not a model measurement, and an invariance statistic computed on an
unstable ranking is a correlation between two noise vectors. So:

  0. ANTI-VACUITY  the readout must have room to vary on BOTH axes before either is interpreted. A
                   pinned readout reads as gloriously "construction-invariant" and means nothing.
  1. SIGNAL        per construction, the across-model spread must exceed the across-seed spread.
  2. STABILITY     the model ranking must reproduce across seeds before any ranking is compared.
  3. INVARIANCE    only then: do different constructions produce the same model ranking?

A NUISANCE GATE, ADDED AFTER THE FACT AND THE NEWEST OF THEM. A readout can pass all three while
being an arithmetic function of something nobody meant to measure. The project's attractor share is
`top1`, the largest token's share of the settled lattice -- and on a small alphabet a lattice that
crystallises into a period-p orbit reads `top1 = 1/p` exactly, whatever the model does. Three of the
remote cells were period-3 and period-4 crystals reading 0.3333 and 0.2500 to the last digit. That
is not noise and no range gate sees it: the values are well separated and perfectly stable. Only
comparing the readout against its value under an explicit nuisance hypothesis catches it, which is
what `nuisance_identity` is for -- and it can only be computed if the run STORED the state, which is
why the protocol requires that.

WHAT THIS MODULE DOES NOT DO. It does not tell you a readout is good. Every gate here can only
downgrade a claim to NOT_DECIDABLE; a readout that passes has cleared necessary conditions across
the constructions you happened to test, which is a statement about your grid, not about the world.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gate import Verdict, DECIDED, NOT_DECIDABLE
from .leverage import LeverageReport, dynamic_range
from .ranking import spearman

__all__ = [
    "Loopness",
    "COMMITMENT_ORDER",
    "DOMAIN_KINDS",
    "DiscriminatorReport",
    "discriminate",
    "nuisance_identity",
    "CONSTRUCTION_DETERMINED",
    "MODEL_DETERMINED",
]

CONSTRUCTION_DETERMINED = "CONSTRUCTION_DETERMINED"
MODEL_DETERMINED = "MODEL_DETERMINED"

# HOW PERMANENT IS A TOKEN ONCE IT IS EMITTED. This is the axis the constructions differ along, and
# naming it is half the contribution: a ring revokes every commitment each sweep, a diffusion
# schedule revokes some of them, speculative decoding revokes on rollback, and ordinary
# autoregressive generation revokes none -- which is why healing exists in the first and nowhere in
# deployment. Ordered loosest-to-tightest so a sweep along it is a sweep in one direction.
COMMITMENT_ORDER = ("in_place", "scheduled", "rollback", "append_only", "free_ar")

# WHAT PRECEDES THE STATE. Added because it turned out to move a readout further than any parameter
# already in this vector. In the project this package came from, the fixed-point class of a greedy
# map was measured with nothing before the state and again behind each model's own chat template:
# nine template tokens took one model's fixed-point fraction from 0.948 to 0.000 and changed its
# class, while eleven tokens took another's from 0.615 to 0.844 and did not -- same weights, same
# estimator, same seeds. An earlier result in the same project had already moved a frozen fraction
# from 74.4% to 24.1% on a SINGLE bos token.
#
# Two consequences for anyone using this protocol. The domain is a CONSTRUCTION parameter, so two
# runs differing only in it are two constructions and the discriminator will treat them as such.
# And the effect was model-specific in direction -- destroying structure in one model, reinforcing
# it in another -- so it cannot be calibrated away with a correction factor; it has to be varied.
DOMAIN_KINDS = ("raw", "bos", "system_prompt", "chat_template", "few_shot", "custom")


@dataclass(frozen=True)
class Loopness:
    """One point on the axis from a closed self-feeding loop to free autoregressive generation.

    The parameters that make a loop a loop, written down instead of left implicit in a script:

      radius       how many neighbours a site conditions on; None means the whole prefix
      temperature  the sampling temperature, or None if the loop is deterministic
      scheme       "sync" | "async" | "ordered" | "none" -- the visit order. Not cosmetic: with
                   asynchronous updating the within-sweep reach is set by visit order, not by the
                   radius, and the project's own cone bound was wrong by ~6x for assuming otherwise.
      commitment   one of COMMITMENT_ORDER; see the note above.
      masking      whatever masking policy the construction applies, as a free-text label.
      domain       what CONDITIONS the state before the loop sees it: one of DOMAIN_KINDS. "raw"
                   means nothing precedes it. This is the axis most likely to be left implicit and
                   least safe to leave implicit -- see the note on DOMAIN_KINDS above.
      domain_tokens how many tokens that conditioning occupies. Recorded separately from the kind
                   because the SIZE is the measurable quantity: one bos token and a thirty-token
                   chat template are both "a prefix" and are not the same construction.

    `label` is what appears in reports. Two Loopness values that differ in any field are different
    constructions, and the whole point of the discriminator is that a readout is allowed to differ
    between them.
    """

    radius: int | None = None
    temperature: float | None = None
    scheme: str = "async"
    commitment: str = "in_place"
    masking: str = "none"
    domain: str = "raw"
    domain_tokens: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.commitment not in COMMITMENT_ORDER:
            raise ValueError(
                f"commitment {self.commitment!r} is not one of {COMMITMENT_ORDER}; add it to "
                "COMMITMENT_ORDER with an explicit position rather than passing a free string, "
                "because the position is what makes a gradient sweep meaningful")
        if self.scheme not in ("sync", "async", "ordered", "none"):
            raise ValueError(f"scheme {self.scheme!r} is not a recognised visit order")
        if self.domain not in DOMAIN_KINDS:
            raise ValueError(
                f"domain {self.domain!r} is not one of {DOMAIN_KINDS}; use 'custom' and set "
                "domain_tokens rather than inventing a label, so the axis stays comparable")
        if self.domain == "raw" and self.domain_tokens:
            raise ValueError("domain='raw' means nothing precedes the state, so domain_tokens "
                             "must be 0; a nonzero prefix is a different domain")
        if self.domain != "raw" and not self.domain_tokens:
            raise ValueError(
                f"domain={self.domain!r} with domain_tokens=0 records a prefix of unknown size. "
                "The size is the measurable quantity -- one bos token and a thirty-token template "
                "are not the same construction -- so it must be stated")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        bits = [f"r{self.radius}" if self.radius is not None else "rfull"]
        if self.temperature is not None:
            bits.append(f"T{self.temperature:g}")
        bits += [self.scheme, self.commitment]
        if self.domain != "raw":                    # raw is the default and stays silent
            bits.append(f"{self.domain}{self.domain_tokens}")
        return ".".join(bits)

    @property
    def commitment_rank(self) -> int:
        """Position on the commitment axis: 0 is a ring, len-1 is free generation."""
        return COMMITMENT_ORDER.index(self.commitment)

    def block(self) -> dict:
        return dict(name=self.name, radius=self.radius, temperature=self.temperature,
                    scheme=self.scheme, commitment=self.commitment,
                    commitment_rank=self.commitment_rank, masking=self.masking,
                    domain=self.domain, domain_tokens=self.domain_tokens)


def _grid(observations: Mapping[tuple, float]) -> tuple[list, list, list, dict]:
    models, constructions, seeds = [], [], []
    table: dict[tuple, float] = {}
    for key, value in observations.items():
        if len(key) != 3:
            raise ValueError(
                f"observation key {key!r} is not (model, construction, seed) -- the three-part key "
                "is the whole design: without seeds there is no noise floor and the signal gate "
                "cannot run")
        m, c, s = key
        c = c.name if isinstance(c, Loopness) else c
        if m not in models:
            models.append(m)
        if c not in constructions:
            constructions.append(c)
        if s not in seeds:
            seeds.append(s)
        table[(m, c, s)] = float(value)
    return models, constructions, seeds, table


def _column(table, models, con, seed):
    """The readout across models at one construction and seed; None if any cell is missing."""
    out = []
    for m in models:
        v = table.get((m, con, seed))
        if v is None or not np.isfinite(v):
            return None
        out.append(v)
    return out


def nuisance_identity(values: Sequence[float], predicted: Sequence[float], *,
                      tol: float = 1e-6, max_frac: float = 0.25,
                      name: str = "readout",
                      nuisance: str = "a nuisance property of the state") -> LeverageReport:
    """Is the readout just an arithmetic function of something nobody meant to measure?

    `predicted` is the readout's value under an explicit nuisance hypothesis, computed per cell from
    the STORED STATE -- e.g. 1/period for a share readout on a lattice that may have crystallised
    into a periodic orbit. The gate fails when too large a fraction of cells match that prediction,
    because on those cells the readout carries the nuisance and nothing else.

    This is the one gate in the package that cannot be computed from the reported numbers alone. It
    needs the object the measurement was reduced from, which is why the protocol requires runs to
    store their state: the project's own remote campaign ran to completion, produced stable
    well-separated values, and was reading orbit lengths -- discovered only when the rings were
    finally kept instead of the scalars.
    """
    v = np.asarray(values, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if v.size != p.size or v.size == 0:
        return LeverageReport("nuisance_identity", False,
                              f"{name}: {v.size} values against {p.size} predictions -- the "
                              "nuisance hypothesis must be evaluated on every cell or on none",
                              dict(n=int(v.size), n_predicted=int(p.size)))
    live = np.isfinite(v) & np.isfinite(p)
    if not live.any():
        return LeverageReport("nuisance_identity", False,
                              f"{name}: no cell has both a value and a nuisance prediction",
                              dict(n=int(v.size), n_live=0))
    match = np.abs(v[live] - p[live]) <= tol
    frac = float(match.mean())
    ok = bool(frac <= max_frac)
    return LeverageReport(
        "nuisance_identity", ok,
        f"{name} equals its value under {nuisance} on {match.sum()} of {int(live.sum())} cells "
        f"({frac:.1%}, gate {max_frac:.0%})"
        + ("" if ok else
           " -- on those cells the readout is that property by arithmetic and carries nothing "
           "about the model, and no range or stability gate can see it because the values are "
           "well separated and perfectly reproducible"),
        dict(n_matching=int(match.sum()), n=int(live.sum()), frac=frac, max_frac=max_frac,
             tol=tol))


@dataclass
class DiscriminatorReport:
    """The two-axis result. `verdict` is the only field a caller should branch on."""

    readout: str
    models: list
    constructions: list
    seeds: list
    model_spread: dict            # construction -> across-model spread at seed[0]
    seed_noise: dict              # construction -> mean |seed A - seed B| across models
    construction_spread: dict     # model -> across-construction spread at seed[0]
    signal: list                  # constructions where model_spread >= k * seed_noise
    seed_stability: float | None  # mean Spearman between the per-seed model rankings
    invariance: float | None      # mean pairwise Spearman across signal-carrying constructions
    range_ratio: float | None     # mean construction spread / mean model spread
    gates: list = field(default_factory=list)
    verdict: Verdict = field(default_factory=lambda: Verdict(status=NOT_DECIDABLE))
    thresholds: dict = field(default_factory=dict)

    def block(self) -> dict:
        return dict(
            readout=self.readout, n_models=len(self.models),
            constructions=list(self.constructions), seeds=list(self.seeds),
            model_spread={k: round(v, 6) for k, v in self.model_spread.items()},
            seed_noise={k: round(v, 6) for k, v in self.seed_noise.items()},
            construction_spread={str(k): round(v, 6) for k, v in self.construction_spread.items()},
            signal=list(self.signal), n_signal=len(self.signal),
            seed_stability=self.seed_stability, invariance=self.invariance,
            range_ratio=self.range_ratio, thresholds=dict(self.thresholds),
            gates=[g.block() for g in self.gates],
            status=self.verdict.status, attribution=self.verdict.value,
            reason=self.verdict.reason)

    def summary(self) -> str:
        """One paragraph, in the shape a results file's verdict string wants."""
        bits = [f"{self.readout}: {len(self.signal)}/{len(self.constructions)} constructions carry "
                f"signal above seed noise"]
        if self.seed_stability is not None:
            bits.append(f"seed-stable ranking {self.seed_stability:+.3f}")
        if self.invariance is not None:
            bits.append(f"cross-construction agreement {self.invariance:+.3f}")
        if self.range_ratio is not None:
            bits.append(f"the construction moves it {self.range_ratio:.1f}x more than the model does")
        return "; ".join(bits) + f". {self.verdict.status}"
        # deliberately no adjective: the caller writes the interpretation, the report writes numbers


def discriminate(observations: Mapping[tuple, float], *, readout: str = "readout",
                 noise_factor: float = 2.0, seed_stability_min: float = 0.6,
                 concordant: float = 0.6, min_models: int = 4,
                 nuisance_prediction: Mapping[tuple, float] | None = None,
                 nuisance_name: str = "a nuisance property of the state") -> DiscriminatorReport:
    """Run the two-axis test on a (model, construction, seed) -> value grid.

    Returns a report whose `verdict` is one of:

      MODEL_DETERMINED         signal, a stable ranking, and that ranking survives construction
                               change. The readout is about the model ACROSS THE CONSTRUCTIONS
                               TESTED -- never in general.
      CONSTRUCTION_DETERMINED  signal and a stable ranking, but different constructions rank the
                               models differently. The readout is about the apparatus.
      NOT_DECIDABLE            the grid could not answer: the readout is pinned on an axis, model
                               identity does not move it beyond seed noise, or the spread carries no
                               ranking the readout can reproduce.

    The last of those is the one worth having. A spread that exceeds noise while producing no
    reproducible ordering is neither a model measurement nor a construction measurement, and the
    honest report is that it is neither.
    """
    models, constructions, seeds, table = _grid(observations)
    thresholds = dict(noise_factor=noise_factor, seed_stability_min=seed_stability_min,
                      concordant=concordant, min_models=min_models)
    empty = DiscriminatorReport(readout, models, constructions, seeds, {}, {}, {}, [],
                                None, None, None, thresholds=thresholds)
    if len(seeds) < 2:
        empty.verdict = Verdict(NOT_DECIDABLE, reason=(
            f"{len(seeds)} seed(s): with one seed there is no noise floor, so 'the models differ' "
            "cannot be distinguished from 'the seeds differ' and no gate below can run"))
        return empty
    if len(constructions) < 2:
        empty.verdict = Verdict(NOT_DECIDABLE, reason=(
            f"{len(constructions)} construction(s): the construction axis is the test, and with one "
            "construction there is nothing to vary it against"))
        return empty
    if len(models) < min_models:
        empty.verdict = Verdict(NOT_DECIDABLE, reason=(
            f"{len(models)} models against a floor of {min_models}: a rank correlation on fewer "
            "points takes a handful of discrete values and cannot fail informatively -- which is "
            "this package's own defect class, arriving inside the discriminator"))
        return empty

    s0, s1 = seeds[0], seeds[1]
    model_spread, seed_noise, sig = {}, {}, []
    for c in constructions:
        a, b = _column(table, models, c, s0), _column(table, models, c, s1)
        if a is None or b is None:
            continue
        model_spread[c] = float(max(a) - min(a))
        seed_noise[c] = float(np.mean([abs(x - y) for x, y in zip(a, b)]))
        if seed_noise[c] > 0 and model_spread[c] >= noise_factor * seed_noise[c]:
            sig.append(c)
    construction_spread = {}
    for m in models:
        vals = [table[(m, c, s0)] for c in constructions if (m, c, s0) in table]
        if len(vals) >= 2:
            construction_spread[m] = float(max(vals) - min(vals))

    mean_model = float(np.mean(list(model_spread.values()))) if model_spread else None
    mean_con = float(np.mean(list(construction_spread.values()))) if construction_spread else None
    ratio = (mean_con / mean_model) if (mean_model and mean_con is not None) else None

    # GATE 0, ANTI-VACUITY, AND IT BELONGS ON THE CONSTRUCTION AXIS ONLY.
    #
    # The model axis is already gated, harder and per construction, by the SIGNAL step below -- so
    # duplicating it here does nothing except steal the informative branch: a first version of this
    # function pre-empted "signal on 2 of 4 constructions" with a global range failure, which is a
    # true statement that tells the caller less. Gate 0 covers the case the signal step structurally
    # cannot see: if varying the CONSTRUCTION does not move the readout, then "the model ranking
    # survives construction change" is true because nothing changed, and INVARIANCE passes
    # vacuously. That is the pinned-observable failure this gate exists for.
    gates = []
    floor = float(np.mean(list(seed_noise.values()))) if seed_noise else 0.0
    if construction_spread:
        gates.append(dynamic_range(
            list(construction_spread.values()) + [0.0], floor=floor, k=noise_factor,
            name=f"{readout} across constructions (largest within-model span)"))
    if nuisance_prediction is not None:
        keys = [k for k in table if k in nuisance_prediction]
        gates.append(nuisance_identity([table[k] for k in keys],
                                       [nuisance_prediction[k] for k in keys],
                                       name=readout, nuisance=nuisance_name))

    rep = DiscriminatorReport(readout, models, constructions, seeds, model_spread, seed_noise,
                              construction_spread, sig, None, None, ratio, gates=gates,
                              thresholds=thresholds)

    failed = [g for g in gates if not g.usable]
    if failed:
        rep.verdict = Verdict(NOT_DECIDABLE,
                              reason="; ".join(f"{g.kind}: {g.reason}" for g in failed))
        return rep

    # STEP 1: SIGNAL, on a majority of constructions.
    if len(sig) <= len(model_spread) / 2:
        rep.verdict = Verdict(NOT_DECIDABLE, reason=(
            f"signal on only {len(sig)} of {len(model_spread)} constructions: model identity does "
            f"not move {readout} beyond seed noise on a majority of the grid, so there is nothing "
            f"for the construction axis to be tested against"))
        return rep

    # STEP 2: SEED STABILITY, before any ranking is compared to any other ranking.
    ag = [spearman(_column(table, models, c, s0), _column(table, models, c, s1))
          for c in sig]
    ag = [v for v in ag if np.isfinite(v)]
    rep.seed_stability = round(float(np.mean(ag)), 4) if ag else None
    if rep.seed_stability is None or rep.seed_stability < seed_stability_min:
        rep.verdict = Verdict(NOT_DECIDABLE, reason=(
            f"seed-stable ranking {rep.seed_stability} against a floor of {seed_stability_min}: "
            f"there is a real spread and no usable ranking inside it. A spread you cannot rank is "
            f"not a model measurement, and an invariance statistic computed on it would be a "
            f"correlation between two noise vectors"))
        return rep

    # STEP 3: INVARIANCE, asked only now.
    ps = [spearman(_column(table, models, x, s0), _column(table, models, y, s0))
          for x, y in itertools.combinations(sig, 2)]
    ps = [v for v in ps if np.isfinite(v)]
    rep.invariance = round(float(np.mean(ps)), 4) if ps else None
    if rep.invariance is None:
        rep.verdict = Verdict(NOT_DECIDABLE, reason=(
            "fewer than two signal-carrying constructions produced a comparable ranking"))
        return rep
    if rep.invariance >= concordant:
        rep.verdict = Verdict(DECIDED, value=MODEL_DETERMINED, reason=(
            f"signal on {len(sig)}/{len(model_spread)} constructions, seed-stable ranking "
            f"{rep.seed_stability:+.3f}, and cross-construction agreement {rep.invariance:+.3f} "
            f">= {concordant}: different constructions rank the models the same way, so {readout} "
            f"is model-attributable ACROSS THESE CONSTRUCTIONS"))
    else:
        rep.verdict = Verdict(DECIDED, value=CONSTRUCTION_DETERMINED, reason=(
            f"signal on {len(sig)}/{len(model_spread)} constructions and a seed-stable ranking "
            f"{rep.seed_stability:+.3f}, but cross-construction agreement {rep.invariance:+.3f} "
            f"< {concordant}: each construction produces its own model ordering, so {readout} is a "
            f"property of the apparatus"
            + (f" -- and the construction moves it {ratio:.1f}x more than the model does"
               if ratio else "")))
    return rep
