"""Frozen per-series MIC rules for pure cultures and retrospective BALF."""
from __future__ import annotations

from collections import Counter
import math
import re

import pandas as pd


def number(value):
    return "" if value is None else f"{value:g}"


def endpoint(concentrations, calls, control):
    if (not concentrations or len(concentrations) != len(calls)
            or any(not math.isfinite(c) or c <= 0 for c in concentrations)
            or any(a >= b for a, b in zip(concentrations, concentrations[1:]))):
        raise ValueError("Expected distinct ascending positive concentrations")
    if any(call not in (0, 1, None) for call in calls) or control not in (0, 1, None, "mixed"):
        raise ValueError("Expected binary calls or explicit missing/discordant replicates")
    growth = [i for i, call in enumerate(calls) if call == 1]
    last = growth[-1] if growth else None
    skipped = [c for i, (c, call) in enumerate(zip(concentrations, calls)) if last is not None and i < last and call == 0]
    result = {"display": "NR", "operator": "", "value": "", "rule": "",
              "skips": len(skipped) if None not in calls else ""}
    if control != 1:
        result["rule"] = {0: "control_no_growth", None: "matched_control_missing", "mixed": "control_replicates_discordant"}[control]
    elif None in calls:
        result["rule"] = "exposed_replicates_discordant"
    elif len(skipped) > 1:
        result["rule"] = "multiple_skipped_concentrations"
    else:
        if last is None:
            op, value, rule = "<=", concentrations[0], "all_exposed_no_growth_lower_limit"
        elif last == len(concentrations) - 1:
            op, value, rule = ">", concentrations[-1], "highest_tested_growth_upper_limit"
        else:
            op, value, rule = "=", concentrations[last + 1], "next_measured_after_highest_growth"
        result.update(display=("" if op == "=" else op) + number(value), operator=op, value=value,
                      rule=("single_skip_" if skipped else "") + rule)
    return result


def parse_reference(token):
    raw = str(token).strip()
    if raw.lower() in {"", "na", "nan", "none", "nr"}:
        return None
    match = re.fullmatch(r"(<=|>=|≤|≥|<|>|=)?\s*([+]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)", raw)
    if not match or not math.isfinite(float(match[2])) or float(match[2]) <= 0:
        raise ValueError(f"Invalid reference MIC: {raw!r}")
    return {"≤": "<=", "≥": ">="}.get(match[1], match[1]) or "=", float(match[2])


def consensus(values):
    return None if not values else (values[0] if len(set(values)) == 1 else "mixed")


def pure_series(frame):
    controls = frame.loc[frame.concentration_ug_ml.eq(0)]
    exposed = frame.loc[frame.concentration_ug_ml.gt(0)]
    observed = consensus(controls.growth.tolist())
    predicted = consensus(controls.prediction.tolist())
    grouped = list(exposed.groupby("concentration_ug_ml", sort=True))
    calls = [consensus(rows.prediction.tolist()) for _, rows in grouped]
    result = endpoint([float(c) for c, _ in grouped], [None if c == "mixed" else c for c in calls], predicted)
    reasons = []
    for prefix, state in (("observed_control", observed), ("predicted_control", predicted)):
        if state != 1:
            reasons.append(prefix + "_" + {None: "missing", "mixed": "replicates_discordant", 0: "no_growth"}[state])
    if "mixed" in calls:
        reasons.append("predicted_exposed_replicates_discordant")
    if result["skips"] not in ("", 0, 1):
        reasons.append("multiple_skipped_concentrations")
    reportable = result["display"] != "NR" and observed == 1
    references = {parse_reference(value) for value in frame.reference_mic}
    references.discard(None)
    if len(references) > 1:
        raise ValueError("Conflicting independent reference MICs within one series")
    reference = next(iter(references)) if references else None
    exclusions = []
    if not reportable:
        exclusions.append("prediction_not_reportable")
    elif result["operator"] != "=":
        exclusions.append("prediction_censored")
    if reference is None:
        exclusions.append("reference_mic_missing")
    elif reference[0] != "=":
        exclusions.append("reference_mic_censored")
    delta = math.log2(result["value"] / reference[1]) if not exclusions else None
    return {"predicted_mic": result["display"] if reportable else "NR",
            "predicted_status": result["operator"] if reportable else "NR",
            "predicted_reportable": reportable, "prediction_rule": result["rule"],
            "prediction_skips": result["skips"], "prediction_NR_reasons": ";".join(reasons),
            "reference_mic": ("" if reference[0] == "=" else reference[0]) + number(reference[1]) if reference else "",
            "numeric_comparison_exclusions": ";".join(exclusions), "numeric_comparable": not exclusions,
            "exact_agreement": abs(delta) <= 1e-12 if delta is not None else "",
            "within_one_twofold": abs(delta) <= 1 + 1e-12 if delta is not None else "",
            "log2_predicted_over_reference": delta,
            "n_wells": len(frame), "n_control_wells": len(controls),
            "observed_control_state": "missing" if observed is None else str(observed),
            "predicted_control_state": "missing" if predicted is None else str(predicted),
            "tested_concentrations_ug_ml": ";".join(number(c) for c, _ in grouped),
            "exposed_prediction_sequence": ";".join(map(str, calls)),
            "row_ids": ";".join(frame.row_id)}


def compare_offscale(test_value, test_sign, reference_value, reference_sign):
    """Paper's frozen CLSI BIT O7/Q7 rule; bounds remain bounds in output."""
    if test_sign not in {"=", "<=", ">", ">="} or reference_sign not in {"=", "<=", ">", ">="}:
        raise ValueError("Unsupported MIC qualifier")
    if not all(math.isfinite(v) and v > 0 for v in (test_value, reference_value)):
        raise ValueError("MICs must be finite and positive")
    test = test_value * (2 if test_sign == ">" else 1)
    reference = reference_value * (2 if reference_sign == ">" else 1)
    tlog, rlog = round(math.log2(test), 1), round(math.log2(reference), 1)
    zero = ((test_sign == "<=" and reference_sign == "<=")
            or (test_sign in {">", ">="} and reference_sign == ">")
            or (test_sign == "<=" and rlog < tlog) or (reference_sign == "<=" and tlog < rlog)
            or (reference_sign in {">", ">="} and tlog > rlog)
            or (test_sign in {">", ">="} and rlog > tlog))
    difference = 0.0 if zero else round(tlog - rlog, 1)
    ea = ((reference_sign == ">" and test >= reference)
          or (test_sign in {">", ">="} and reference > test)
          or (reference_sign == "<=" and test < reference)
          or (test_sign == "<=" and reference < test) or abs(tlog - rlog) <= 1)
    return difference, ea


def balf_series(frame, comparator):
    votes = []
    for concentration, wells in frame.groupby("concentration_ug_ml", sort=True):
        if len(wells) != 3:
            raise ValueError("BALF requires three wells per measured concentration")
        votes.append({"group_id": frame.group_id.iloc[0], "drug_context": frame.drug_context.iloc[0],
                      "concentration_ug_ml": float(concentration), "n_wells": 3,
                      "observed_growth_n": int(wells.growth.sum()),
                      "predicted_growth_n": int(wells.prediction.sum()),
                      "observed_majority": int(wells.growth.sum() >= 2),
                      "prediction_majority": int(wells.prediction.sum() >= 2),
                      "row_ids": ";".join(wells.row_id)})
    if [r["concentration_ug_ml"] for r in votes] != [0, .5, 1, 2, 4, 6, 8]:
        raise ValueError("BALF concentration series changed")
    control, *exposed = votes
    result = endpoint([r["concentration_ug_ml"] for r in exposed],
                      [r["prediction_majority"] for r in exposed], control["prediction_majority"])
    reasons = []
    if not control["observed_majority"]:
        reasons.append("observed_control_no_growth")
    if result["display"] == "NR":
        reasons.append("prediction_" + result["rule"])
    reportable = not reasons
    reference = parse_reference(comparator.reference_mic)
    comparable = reportable and reference is not None
    delta, ea = compare_offscale(result["value"], result["operator"], reference[1], reference[0]) if comparable else ("", "")
    return {"predicted_mic_as_reported": result["display"].replace("<=", "≤") if reportable else "NR",
            "prediction_mic_rule": result["rule"], "skipped_concentration_count": result["skips"],
            "predicted_mic_reportable": reportable, "not_reportable_reasons": ";".join(reasons),
            "hospital_mic_operator": reference[0] if reference else "",
            "hospital_mic_calculation_value_ug_ml": reference[1] if reference else "",
            "comparable_under_author_rule": comparable,
            "offscale_adjusted_dilution_difference": delta,
            "exact_agreement_after_offscale_adjustment": delta == 0 if comparable else "",
            "essential_agreement_after_offscale_adjustment": ea,
            "reference_mic": comparator.reference_mic, "participant_group": comparator.participant_group,
            "n_wells": len(frame), "observed_control_growth_n": control["observed_growth_n"],
            "predicted_control_growth_n": control["predicted_growth_n"],
            "prediction_majority_sequence_including_control": ";".join(str(r["prediction_majority"]) for r in votes),
            "row_ids": ";".join(frame.row_id)}, votes


def mic_tables(development, balf, comparators):
    pure = pd.DataFrame([{"group_id": group, "drug_context": drug, **pure_series(frame)}
                         for (group, drug), frame in development.groupby(["group_id", "drug_context"], sort=True)])
    comparisons = comparators.set_index(["group_id", "drug_context"])
    if set(comparisons.index) != set(balf.groupby(["group_id", "drug_context"]).groups):
        raise ValueError("All BALF samples must retain one comparator record")
    pairs, votes = [], []
    for (group, drug), frame in balf.groupby(["group_id", "drug_context"], sort=True):
        pair, sample_votes = balf_series(frame, comparisons.loc[group, drug])
        pairs.append({"group_id": group, "drug_context": drug, **pair})
        votes.extend(sample_votes)
    pairs = pd.DataFrame(pairs)
    selected = pure.loc[pure.numeric_comparable]
    included = pairs.loc[pairs.comparable_under_author_rule]
    summary = {"development": {"series": len(pure), "reportable": int(pure.predicted_reportable.sum()),
                "status_counts": dict(Counter(pure.predicted_status)), "numeric_pairs": len(selected),
                "exact_n": int(selected.exact_agreement.sum()), "ea_n": int(selected.within_one_twofold.sum())},
               "balf": {"samples": len(pairs), "participant_groups": pairs.participant_group.nunique(),
                "reportable": int(pairs.predicted_mic_reportable.sum()), "comparable": len(included),
                "comparable_participant_groups": included.participant_group.nunique(),
                "exact_after_offscale_n": int(included.exact_agreement_after_offscale_adjustment.sum()),
                "ea_n": int(included.essential_agreement_after_offscale_adjustment.sum())}}
    return pure, pairs, pd.DataFrame(votes), summary
