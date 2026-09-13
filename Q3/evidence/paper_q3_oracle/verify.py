"""Recompute the manuscript's oracle tables from archived data, without simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MC = ROOT / "Q3/results/oracle-tsp-mc-20260912-021030-302657"
COUNTS = ROOT / "Q3/results/official-q3-counts-20260912-022136-803657"
CHAIN = ROOT / "Q3/results/oracle-chain-20260912-022300-858289"


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-8):
        raise ValueError(f"Numeric mismatch: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, help="Optional archived worlds_and_exact_routes.npz")
    parser.add_argument("--output", type=Path, help="Save the verification JSON")
    args = parser.parse_args()
    paths = [MC / "summary.json", MC / "batch.json", MC / "verification.json",
             COUNTS / "summary.json", CHAIN / "summary.json"]
    summary, batch, prior, counts, chain = [json.loads(p.read_text()) for p in paths]
    assert batch["status"] == "completed"
    assert batch["completed_worlds"] == batch["planned_worlds"] == 100000
    assert batch["failed_worlds"] == 0 and prior["status"] == "passed"
    assert summary["all_worlds_completed"]
    groups = [(n, summary["by_N"][str(n)]) for n in range(10, 17)]
    total_maps = sum(g["worlds"] for _, g in groups)
    total_sources = sum(n * g["worlds"] for n, g in groups)
    assert total_maps == 100000 and total_sources == 1299298
    mean_length = sum(g["worlds"] * g["length_m"]["mean"] for _, g in groups) / total_maps
    direct_case = sum(g["worlds"] * g["s_per_source"]["mean"] for _, g in groups) / total_maps
    direct_pooled = mean_length * total_maps / (5 * total_sources)
    close(mean_length, summary["mean_route_length_m"]["mean"])
    close(direct_case, summary["primary_expected_T_div_N_s_per_source"]["mean"])
    close(direct_pooled, summary["pooled_T_div_total_N_s_per_source"]["estimate"])

    official_counts = {f["source_count"]: f["cases"] for f in counts["frequencies"]}
    assert sum(official_counts.values()) == counts["official_Q3_practice_cases"] == 152
    assert sum(n * m for n, m in official_counts.items()) == counts["total_sources"] == 2000
    assert [official_counts[n] for n, _ in groups] == chain["official_N_counts_10_to_16"]

    def reweight(weights):
        case_equal = sum(weights[n] * g["s_per_source"]["mean"] for n, g in groups) / sum(weights.values())
        pooled = sum(weights[n] * n * g["s_per_source"]["mean"] for n, g in groups) / sum(n * weights[n] for n, _ in groups)
        return case_equal, pooled

    uniform = reweight({n: 1 for n, _ in groups})
    empirical = reweight(official_counts)
    close(uniform[0], summary["uniform_N_stratified_estimate"]["expected_case_T_div_N"]["mean_s_per_source"])
    close(uniform[1], summary["uniform_N_stratified_estimate"]["expected_T_div_expected_N"]["mean_s_per_source"])
    close(empirical[0], chain["oracle_official152_N_reweighted_case_equal_s_per_source"])
    close(empirical[1], chain["oracle_official152_N_reweighted_pooled_s_per_source"])

    paper = (ROOT / "essay/essay.tex").read_text()
    by_n_table = paper.split(r"\label{tab:q3-oracle-by-n}", 1)[1].split(r"\end{table}", 1)[0]
    for n, g in groups:
        close(g["length_m"]["mean"] / (5 * n), g["s_per_source"]["mean"])
        row = f'{n} & {g["worlds"]} & {g["length_m"]["mean"]:.2f} & {g["s_per_source"]["mean"]:.2f}'
        assert row in by_n_table, row
    weighted_table = paper.split(r"\label{tab:q3-oracle-weights}", 1)[1].split(r"\end{table}", 1)[0]
    comparisons = [("十万图直接统计", (direct_case, direct_pooled)),
                   ("各源数按 $1/7$ 分层加权", uniform),
                   ("按 $152$ 局演练源数频率加权", empirical)]
    for label, (case, pooled) in comparisons:
        row = f"{label} & {case:.4f} & {pooled:.4f}"
        assert row in weighted_table, row

    result = {
        "status": "passed", "maps": total_maps, "sources": total_sources,
        "by_N": [{"N": n, "maps": g["worlds"], "mean_length_m": g["length_m"]["mean"],
                  "mean_movement_s_per_source": g["s_per_source"]["mean"]} for n, g in groups],
        "direct_case_equal_s_per_source": direct_case,
        "direct_pooled_s_per_source": direct_pooled,
        "uniform_stratified_case_equal_and_pooled": uniform,
        "official152_case_equal_and_pooled": empirical,
        "official152_pooled_plus_clear_s_per_source": empirical[1] + 5,
        "all_ten_table_rows_match_archived_data": True,
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "limits": "Independent random maps; movement only; 20m disk optimum not solved; no paired strategy comparison.",
    }

    if args.routes:
        import numpy as np
        from scipy.stats import t
        with np.load(args.routes) as arrays:
            ns, xy, order, lengths = [arrays[k] for k in
                                     ("source_counts", "source_xy", "optimal_order", "optimal_length_m")]
        assert len(ns) == total_maps and int(ns.sum()) == total_sources
        largest_error = 0.0
        for n, g in groups:
            mask = ns == n
            assert int(mask.sum()) == g["worlds"]
            points, permutation = xy[mask, :n], order[mask, :n]
            assert np.all(np.sort(permutation, axis=1) == np.arange(n))
            assert np.all(np.linalg.norm(points, axis=2) <= 1800 + 1e-9)
            route = np.take_along_axis(points, permutation[:, :, None], axis=1)
            actual = np.linalg.norm(route[:, 0], axis=1) + np.linalg.norm(np.diff(route, axis=1), axis=2).sum(axis=1)
            largest_error = max(largest_error, float(np.max(np.abs(actual - lengths[mask]))))
            close(float(lengths[mask].mean()), g["length_m"]["mean"])
        assert largest_error < 1e-8
        per_source = lengths / (5 * ns)
        close(float(per_source.mean()), direct_case)
        se = float(per_source.std(ddof=1) / np.sqrt(total_maps))
        expected = summary["primary_expected_T_div_N_s_per_source"]
        close(se, expected["mc_standard_error"])
        margin = float(t.ppf(0.975, total_maps - 1)) * se
        for actual, target in zip([per_source.mean() - margin, per_source.mean() + margin], expected["mean_95_interval"]):
            close(float(actual), target)
        lower = np.maximum(0, lengths - 20 * (2 * ns - 1) - 0.01) / (5 * ns)
        close(float(lower.mean()), summary["expected_20m_visit_bound_s_per_source"]["lower_bound_mean"])
        result["archived_routes"] = {
            "file": args.routes.name, "sha256": hashlib.sha256(args.routes.read_bytes()).hexdigest(),
            "checked_routes": len(ns), "maximum_length_error_m": largest_error,
            "all_routes_visit_each_source_once": True, "all_positions_inside_disk": True,
            "case_equal_mc_standard_error": se,
            "case_equal_20m_visit_geometric_bounds": [float(lower.mean()), float(per_source.mean())],
            "note": "Routes and statistics rechecked; optimality independently checked in archived verification.json.",
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
