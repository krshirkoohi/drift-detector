"""
experiments/pollutant_validation.py
====================================
Off-topic Pollutant Validation Experiment
------------------------------------------
Runs the length × pollutant-rate grid across three pollution arms and
measures whether the drift detector flags contamination before semantic
similarity (probe accuracy) falls below the baseline threshold.

Grid
----
  Session lengths  : 5, 10, 15 turns
  Pollution rates  : 0%, 33%, 66%, 100%
  Pollutant arms   : random_sentences | topic_switched | factually_wrong

Total cells: 3 lengths × 4 rates × 3 arms = 36

For each cell the script:
  1. Builds a turn sequence mixing clean + polluted responses at the
     requested rate (random shuffle preserving proportion).
  2. Feeds every turn through AgentHarness / DriftDetector (local
     roberta-base adapter — no API key required).
  3. Records per-cell metrics:
       - mean_similarity   : mean cosine similarity to centroid (1 - dist)
       - mean_drift_score  : mean cosine distance
       - alarmed           : True if detector fired at any turn
       - alarm_turn        : index of first alarm (None if no alarm)
       - accuracy_gap      : similarity drop vs. clean (0%) arm of same length
       - dod_pass          : detector alarmed BEFORE accuracy fell below threshold

Acceptance test (DoD)
----------------------
  For each polluted cell where accuracy_gap > 0:
    The detector must have alarmed, AND the alarm must have fired at or
    before the turn when similarity first dropped below the clean arm's
    mean similarity.

Usage
-----
    python experiments/pollutant_validation.py
    python experiments/pollutant_validation.py --metric euclidean
    python experiments/pollutant_validation.py --out results/pollutant_grid.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

# ── path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from drift_detector.baseline import BaselineStore
from drift_detector.detector import DriftDetector
from drift_detector.embeddings import LocalEmbeddingAdapter
from drift_detector.harness import AgentHarness

# ── grid constants ───────────────────────────────────────────────────────────
LENGTHS       = [5, 10, 15]
RATES         = [0.0, 0.33, 0.66, 1.0]   # proportion of turns that are polluted
ARMS          = ["random_sentences", "topic_switched", "factually_wrong"]
BASELINE_FILE = os.path.join(PROJECT_ROOT, "baselines", "default.json")

# ── fixture corpus ───────────────────────────────────────────────────────────
# Clean responses — on-topic Python / ML-engineering domain
CLEAN: List[str] = [
    "You can sort a Python list in-place using list.sort(), or get a new sorted list with the built-in sorted() function.",
    "Use a dictionary comprehension to invert a mapping: {v: k for k, v in original.items()}.",
    "Python's contextlib.suppress() is a clean way to silently ignore specific exception types without a bare try/except.",
    "The @dataclass decorator auto-generates __init__, __repr__, and __eq__ from annotated class fields.",
    "You can chain itertools.chain() calls to lazily flatten nested iterables without materialising them in memory.",
    "Use functools.lru_cache to memoize expensive pure functions with a simple decorator.",
    "List comprehensions are generally faster than equivalent for-loop appends because the interpreter optimises the bytecode.",
    "Pathlib.Path provides an object-oriented interface to the filesystem that works cross-platform without os.path juggling.",
    "The walrus operator (:=) lets you assign a variable inside an expression, useful for avoiding repeated function calls in loops.",
    "Generator expressions are memory-efficient alternatives to list comprehensions when you only need to iterate once.",
    "asyncio.gather() runs multiple coroutines concurrently and collects their results in order.",
    "Use __slots__ to reduce per-instance memory overhead when creating many instances of a class.",
    "pytest fixtures with the 'function' scope are torn down and re-created between every test.",
    "The bisect module provides O(log n) insertion into a sorted list using binary search.",
    "NumPy broadcasting lets you perform element-wise operations on arrays of different shapes without explicit loops.",
]

# Pollutant arm 1 — random, unrelated sentences
RANDOM_SENTENCES: List[str] = [
    "The migratory patterns of Arctic terns cover roughly 70,000 kilometres per year.",
    "Sourdough fermentation relies on wild yeast and Lactobacillus bacteria in the starter culture.",
    "The circumference of Jupiter at its equator is approximately 439,264 kilometres.",
    "Watercolour pigments achieve transparency by allowing light to reflect from the paper beneath the wash.",
    "The Great Barrier Reef spans over 2,300 kilometres along the Queensland coastline.",
    "A standard espresso shot is extracted at nine bars of pressure for about 25 to 30 seconds.",
    "The Treaty of Westphalia in 1648 established the principle of state sovereignty in international relations.",
    "Tectonic plate movement averages between two and fifteen centimetres per year depending on the boundary type.",
    "Bioluminescence in deep-sea organisms is produced by a chemical reaction between luciferin and luciferase.",
    "The Maillard reaction between amino acids and reducing sugars produces the flavour and colour of browned food.",
    "Gregorian chant developed in medieval Europe as the liturgical music of the Roman Catholic Church.",
    "The human femur is the longest and strongest bone in the body.",
    "Venice is built on 118 small islands connected by a network of canals and bridges.",
    "The Fibonacci sequence appears in phyllotaxis, the arrangement of leaves and seeds in plants.",
    "Carbon fibre composites are used in aerospace because of their high strength-to-weight ratio.",
]

# Pollutant arm 2 — topic-switched: plausible but wrong domain (history / cooking / geography)
TOPIC_SWITCHED: List[str] = [
    "The best way to caramelise onions is low heat with a pinch of salt over 40 minutes; rushing it burns them.",
    "Napoleon's decisive defeat at Waterloo in 1815 ended the Napoleonic Wars and led to his exile on Saint Helena.",
    "To prepare a roux, cook equal parts butter and flour over medium heat until it smells nutty and turns golden.",
    "The Silk Road connected China to the Mediterranean, facilitating trade in textiles, spices, and ideas.",
    "Sautéing vegetables in a hot, dry pan before adding liquid builds depth of flavour through the Maillard reaction.",
    "The Roman Forum was the centre of political, religious, and commercial life in ancient Rome.",
    "A good vinaigrette uses a three-to-one ratio of oil to acid, emulsified with a small amount of mustard.",
    "The Renaissance began in Florence in the 14th century, driven by wealthy patrons like the Medici family.",
    "Blanching vegetables in salted boiling water and then shocking them in ice water preserves colour and texture.",
    "The Amazon River discharges roughly 20% of all fresh water entering the world's oceans.",
    "Braising is a two-step process: sear the protein for colour, then slow-cook it in liquid to break down collagen.",
    "The Industrial Revolution began in Britain around 1760, transforming manufacturing through steam power and machinery.",
    "Umami is the fifth basic taste, caused by glutamate compounds found naturally in mushrooms and aged cheese.",
    "The Berlin Wall fell on the 9th of November 1989, symbolising the end of the Cold War.",
    "Tempering chocolate involves raising and lowering its temperature to stabilise the cocoa butter crystals.",
]

# Pollutant arm 3 — factually wrong Python / ML answers
FACTUALLY_WRONG: List[str] = [
    "Python lists are immutable; once created you cannot add or remove elements without creating a new list.",
    "The range() function in Python 3 returns a list of integers, just like it did in Python 2.",
    "You should always use bare except: clauses to catch all exceptions; it is considered best practice.",
    "Dictionary keys in Python must be integers or strings; tuples and frozensets cannot be used as keys.",
    "The GIL (Global Interpreter Lock) allows Python threads to run in true parallel on multi-core machines.",
    "NumPy arrays are slower than Python lists for numerical operations because of the overhead of C extensions.",
    "Generators are identical to lists in Python; the only difference is the syntax used to create them.",
    "A set in Python maintains insertion order, just like a list does.",
    "Using == to compare floating-point numbers is always reliable in Python because floats are exact.",
    "The json.loads() function writes a Python object to a JSON-formatted string.",
    "Lambda functions in Python can contain multiple statements separated by semicolons.",
    "Decorators in Python permanently modify the source code of the function they are applied to.",
    "pip install always installs packages into the system Python, regardless of whether a virtualenv is active.",
    "asyncio.sleep() blocks the entire event loop, preventing other coroutines from running.",
    "Pandas DataFrames use row-major storage (C order) by default, unlike NumPy arrays which are column-major.",
]

ARM_CORPUS: Dict[str, List[str]] = {
    "random_sentences": RANDOM_SENTENCES,
    "topic_switched":   TOPIC_SWITCHED,
    "factually_wrong":  FACTUALLY_WRONG,
}

# ── result dataclass ─────────────────────────────────────────────────────────
@dataclass
class CellResult:
    arm:             str
    n_turns:         int
    pollution_rate:  float
    n_polluted:      int
    mean_similarity: float   # mean (1 - cosine_distance)
    mean_drift_score: float  # mean cosine_distance
    alarmed:         bool
    alarm_turn:      Optional[int]
    accuracy_gap:    float   # similarity drop vs. 0% arm for same length
    dod_pass:        Optional[bool]  # None when pollution_rate == 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ── session builder ──────────────────────────────────────────────────────────
def build_turn_sequence(
    n_turns: int,
    rate: float,
    arm: str,
    rng: random.Random,
) -> List[Tuple[str, bool]]:
    """
    Return a list of (response_text, is_polluted) tuples.
    Polluted turns are drawn from the arm corpus; clean turns from CLEAN.
    The polluted turns are randomly interspersed.
    """
    n_polluted = round(n_turns * rate)
    n_clean    = n_turns - n_polluted

    clean_pool     = rng.choices(CLEAN, k=n_clean)
    pollutant_pool = rng.choices(ARM_CORPUS[arm], k=n_polluted)

    sequence = [(t, False) for t in clean_pool] + [(t, True) for t in pollutant_pool]
    rng.shuffle(sequence)
    return sequence


# ── single cell runner ───────────────────────────────────────────────────────
def run_cell(
    harness: AgentHarness,
    arm: str,
    n_turns: int,
    rate: float,
    rng: random.Random,
    clean_mean_similarity: Optional[float],
) -> CellResult:
    """Run one grid cell and return a CellResult."""
    turns = build_turn_sequence(n_turns, rate, arm, rng)
    session_id = f"{arm}__n{n_turns}__r{int(rate*100):03d}"

    harness.start_session(session_id=session_id)
    records = []
    for prompt_stub, _ in turns:
        rec = harness.process_turn(
            user_prompt="[probe]",
            agent_response=prompt_stub,
        )
        records.append(rec)
    harness.end_session()

    similarities = [1.0 - r.cosine_distance for r in records]
    drift_scores = [r.cosine_distance for r in records]
    mean_sim     = sum(similarities) / len(similarities)
    mean_drift   = sum(drift_scores) / len(drift_scores)

    # First alarm turn (1-indexed, None if never alarmed)
    alarmed = any(r.is_drifting for r in records)
    alarm_turn = next((r.turn_index for r in records if r.is_drifting), None)

    # Accuracy gap vs clean (0%) arm
    accuracy_gap = (clean_mean_similarity - mean_sim) if clean_mean_similarity is not None else 0.0

    # DoD: did the detector alarm before accuracy dropped below clean threshold?
    if rate == 0.0 or not alarmed:
        dod_pass = None if rate == 0.0 else False
    else:
        # Find the first turn where similarity dropped below the clean arm mean
        drop_turn = next(
            (r.turn_index for r in records if (1.0 - r.cosine_distance) < (clean_mean_similarity or 0.0)),
            None
        )
        if drop_turn is None:
            # Accuracy never dropped below clean threshold — DoD vacuously passes
            dod_pass = True
        else:
            dod_pass = alarm_turn <= drop_turn

    return CellResult(
        arm=arm,
        n_turns=n_turns,
        pollution_rate=rate,
        n_polluted=round(n_turns * rate),
        mean_similarity=mean_sim,
        mean_drift_score=mean_drift,
        alarmed=alarmed,
        alarm_turn=alarm_turn,
        accuracy_gap=accuracy_gap,
        dod_pass=dod_pass,
    )


# ── pretty printer ───────────────────────────────────────────────────────────
def _bar(width: int = 72) -> str:
    return "─" * width

def print_arm_table(arm: str, results: List[CellResult]) -> None:
    arm_results = [r for r in results if r.arm == arm]
    print(f"\n  Arm: {arm.replace('_', ' ').upper()}")
    print(f"  {'Turns':>6}  {'Rate':>6}  {'MeanSim':>8}  {'MeanDist':>9}  "
          f"{'Alarmed':>8}  {'AlarmTurn':>10}  {'AccuGap':>8}  {'DoD':>5}")
    print("  " + _bar(72))
    for r in arm_results:
        dod = ("✅" if r.dod_pass else "❌") if r.dod_pass is not None else " — "
        alarm_str = str(r.alarm_turn) if r.alarm_turn else "—"
        print(
            f"  {r.n_turns:>6}  {r.pollution_rate:>5.0%}  "
            f"{r.mean_similarity:>8.4f}  {r.mean_drift_score:>9.4f}  "
            f"{'YES' if r.alarmed else 'no':>8}  {alarm_str:>10}  "
            f"{r.accuracy_gap:>+8.4f}  {dod:>5}"
        )


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Off-topic Pollutant Validation Experiment")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--seed",   type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile of baseline distances used to auto-calibrate the threshold (default: 99)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the auto-calibrated threshold with a fixed value"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "experiments", "results", "pollutant_grid.json"),
        help="Output path for JSON results"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("\n" + "═" * 72)
    print("  OFF-TOPIC POLLUTANT VALIDATION EXPERIMENT")
    print("  Grid: 3 lengths × 4 rates × 3 arms = 36 cells")
    print(f"  Metric: {args.metric.upper()}  ·  Seed: {args.seed}")
    print("═" * 72)

    # ── initialise detector (local, offline) ─────────────────────────────
    print("\n⚙  Loading roberta-base and computing baseline centroid...")
    t0 = time.time()
    adapter = LocalEmbeddingAdapter("roberta-base")
    store   = BaselineStore(BASELINE_FILE)

    # Auto-calibrate at the requested percentile, then optionally override
    store.compute_centroid(adapter=adapter)
    auto_threshold = store.calculate_percentile_threshold(args.metric, args.percentile)
    threshold = args.threshold if args.threshold is not None else auto_threshold

    detector = DriftDetector(
        baseline_store=store,
        threshold=threshold,
        metric=args.metric,
        use_trend=False,   # threshold-mode for this experiment
        embedding_adapter=adapter,
    )
    harness = AgentHarness(detector=detector, log_dir=None, verbose=False)
    print(
        f"   Done in {time.time()-t0:.1f}s  ·  "
        f"percentile={args.percentile:.0f}th  ·  threshold={detector.threshold:.4f}"
        + ("  (overridden)" if args.threshold is not None else "")
        + "\n"
    )

    # ── run the grid ──────────────────────────────────────────────────────
    all_results: List[CellResult] = []
    total = len(ARMS) * len(LENGTHS) * len(RATES)
    done  = 0

    for arm in ARMS:
        # cache clean (0%) mean similarity per length so we can compute accuracy_gap
        clean_means: Dict[int, float] = {}

        # Run 0% first per length so the gap is available for other rates
        for n_turns in LENGTHS:
            result = run_cell(harness, arm, n_turns, 0.0, rng, None)
            clean_means[n_turns] = result.mean_similarity
            all_results.append(result)
            done += 1
            print(f"  [{done:>2}/{total}] {arm:<22} {n_turns:>2} turns  {0:>3.0%} → sim={result.mean_similarity:.4f}")

        # Then run the remaining rates
        for n_turns in LENGTHS:
            for rate in RATES:
                if rate == 0.0:
                    continue  # already done above
                result = run_cell(harness, arm, n_turns, rate, rng, clean_means[n_turns])
                all_results.append(result)
                done += 1
                alarm_str = f"alarm@t{result.alarm_turn}" if result.alarm_turn else "no alarm"
                dod_str   = ("DoD ✅" if result.dod_pass else "DoD ❌") if result.dod_pass is not None else ""
                print(
                    f"  [{done:>2}/{total}] {arm:<22} {n_turns:>2} turns  {rate:>3.0%}"
                    f" → sim={result.mean_similarity:.4f}  gap={result.accuracy_gap:+.4f}"
                    f"  {alarm_str:<12}  {dod_str}"
                )

    # ── summary tables ────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  RESULTS SUMMARY")
    print("═" * 72)
    for arm in ARMS:
        print_arm_table(arm, all_results)

    # ── false-positive rate (clean arm alarms) ───────────────────────────
    clean_cells   = [r for r in all_results if r.pollution_rate == 0.0]
    clean_alarms  = [r for r in clean_cells if r.alarmed]
    fp_rate       = len(clean_alarms) / len(clean_cells) if clean_cells else 0.0

    print(f"{'─'*72}")
    print(f"  FALSE-POSITIVE RATE (clean arm)  —  {len(clean_alarms)}/{len(clean_cells)} clean cells alarmed  ({fp_rate:.0%})")

    # ── DoD verdict ───────────────────────────────────────────────────────
    dod_cells    = [r for r in all_results if r.dod_pass is not None]
    dod_pass_cnt = sum(1 for r in dod_cells if r.dod_pass)
    dod_fail_cnt = len(dod_cells) - dod_pass_cnt

    print(f"  ACCEPTANCE TEST (DoD)  —  {dod_pass_cnt}/{len(dod_cells)} cells passed")
    if dod_fail_cnt:
        print("\n  Failing cells:")
        for r in dod_cells:
            if not r.dod_pass:
                print(f"    {r.arm:<22} {r.n_turns:>2} turns  {r.pollution_rate:>3.0%}  "
                      f"alarm_turn={r.alarm_turn}  gap={r.accuracy_gap:+.4f}")
    print()

    # ── save JSON results ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output = {
        "experiment": "off_topic_pollutant_validation",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metric": args.metric,
        "seed": args.seed,
        "percentile": args.percentile,
        "threshold": detector.threshold,
        "false_positive_rate": fp_rate,
        "grid": {
            "lengths": LENGTHS,
            "rates": RATES,
            "arms": ARMS,
        },
        "dod_summary": {
            "total_cells": len(dod_cells),
            "pass": dod_pass_cnt,
            "fail": dod_fail_cnt,
            "pass_rate": dod_pass_cnt / len(dod_cells) if dod_cells else 0.0,
        },
        "cells": [r.to_dict() for r in all_results],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {args.out}")
    print()

    sys.exit(0 if dod_fail_cnt == 0 else 1)


if __name__ == "__main__":
    main()
