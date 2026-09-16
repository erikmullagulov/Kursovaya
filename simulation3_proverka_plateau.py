import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import cohen_director_cut as cohen_mod
import mle_with_no_L as em_mod


sigma1_true = 1.0
sigma2_true = 1.0

em_n_init = 10
n_workers = 6

n_replicates = 500

results_path = "simulation_proverka_plateau.csv"
checkpoint_every = 500

crossover_n_values = [100, 150, 200, 250, 300, 400, 500, 700, 1000]
crossover_delta = 3.0
crossover_delta_label = "strong"

plateau_n_values = [1500, 2000, 3000, 5000]
plateau_delta = 0.5
plateau_delta_label = "weak"

p_values = {
    "balanced": 0.5,
    "unbalanced": 0.2
}

SEED_BASE = 950_000_000


def make_seed(scenario_idx, n, p_idx, replicate):
    return (
        SEED_BASE
        + scenario_idx * 100_000_000
        + n * 1_000
        + p_idx * 100
        + (replicate % 100)
        + (replicate // 100) * 100_000
    )


def run_replicate(task):

    (scenario, n, delta_label, delta, p_label, p_true, replicate, seed) = task

    mu1_true = 0.0
    mu2_true = delta * sigma1_true

    x = cohen_mod.create_mixture(
        n=n,
        p=p_true,
        mu1=mu1_true,
        mu2=mu2_true,
        sigma1=sigma1_true,
        sigma2=sigma2_true,
        seed=seed,
    )

    base_row = {
        "scenario": scenario,
        "n": n,
        "delta_label": delta_label,
        "delta": delta,
        "p_label": p_label,
        "p_true": p_true,
        "mu1_true": mu1_true,
        "mu2_true": mu2_true,
        "sigma1_sq_true": sigma1_true**2,
        "sigma2_sq_true": sigma2_true**2,
        "replicate": replicate,
        "seed": seed,
    }

    rows = []

     # Cohen (MM)
    row = dict(base_row)
    row["method"] = "MM"
    t0 = time.perf_counter()
    try:
        result = cohen_mod.cohen_fit_safe(x)
    except Exception as e:
        result = None
        row["error"] = f"{type(e).__name__}: {e}"
    row["time_sec"] = time.perf_counter() - t0
    _fill_result(row, result, p_true, mu1_true, mu2_true)
    rows.append(row)

     # EM (MLE)
    row = dict(base_row)
    row["method"] = "MLE"
    t0 = time.perf_counter()
    try:
        result = em_mod.em_fit_safe(x, n_init=em_n_init, seed=seed)
    except Exception as e:
        result = None
        row["error"] = f"{type(e).__name__}: {e}"
    row["time_sec"] = time.perf_counter() - t0
    _fill_result(row, result, p_true, mu1_true, mu2_true)
    rows.append(row)

    return rows


def _fill_result(row, result, p_true, mu1_true, mu2_true):
    if result is None:
        row["success"] = False
        for key in ["p_hat", "mu1_hat", "mu2_hat", "sigma1_sq_hat", "sigma2_sq_hat",
                    "mu1_abs_error", "mu2_abs_error"]:
            row[key] = np.nan
        return

    row["success"] = True
    row["p_hat"] = result["p"]
    row["mu1_hat"] = result["mu1"]
    row["mu2_hat"] = result["mu2"]
    row["sigma1_sq_hat"] = result["sigma1_sq"]
    row["sigma2_sq_hat"] = result["sigma2_sq"]
    row["mu1_abs_error"] = abs(result["mu1"] - mu1_true)
    row["mu2_abs_error"] = abs(result["mu2"] - mu2_true)


def build_tasks():
    tasks = []

    for n in crossover_n_values:
        for p_idx, (p_label, p_true) in enumerate(p_values.items()):
            for replicate in range(n_replicates):
                seed = make_seed(1, n, p_idx, replicate)
                tasks.append((
                    "crossover", n, crossover_delta_label, crossover_delta,
                    p_label, p_true, replicate, seed
                ))

    for n in plateau_n_values:
        for p_idx, (p_label, p_true) in enumerate(p_values.items()):
            for replicate in range(n_replicates):
                seed = make_seed(2, n, p_idx, replicate)
                tasks.append((
                    "plateau", n, plateau_delta_label, plateau_delta,
                    p_label, p_true, replicate, seed
                ))

    return tasks


def main():
    tasks = build_tasks()
    total_tasks = len(tasks)

    all_rows = []
    start_time = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_replicate, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception:
                print(f"UNEXPECTED FAILURE at task={task}")
                traceback.print_exc()
                continue

            completed += 1

            if completed % checkpoint_every == 0 or completed == total_tasks:
                pd.DataFrame(all_rows).to_csv(results_path, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(results_path, index=False)

    a = df[(df.scenario == "crossover") & (df.method.isin(["MM", "MLE"]))]
    b = df[(df.scenario == "plateau") & (df.method == "MM")]


if __name__ == "__main__":
    main()
