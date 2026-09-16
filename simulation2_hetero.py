import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import cohen_director_cut as cohen_mod
import mle_with_no_L as em_mod


n_values = [50, 300, 1000]


separations = {
    "weak": 0.5,
    "medium": 1.5,
    "strong": 3.0,
}

p_values = {
    "balanced": 0.5,
    "unbalanced": 0.2,
}

variance_ratios = {
    "ratio_2": 2.0,
    "ratio_4": 4.0,
    "ratio_9": 9.0,
}

n_replicates = 200

sigma1_true = 1.0

em_n_init = 10
n_workers = 6

results_path = "simulation_results_hetero.csv"
checkpoint_every = 500


def make_seed(n_idx, delta_idx, p_idx, ratio_idx, replicate):
    return (
        900_000_000
        + n_idx * 10_000_000
        + delta_idx * 1_000_000
        + p_idx * 100_000
        + ratio_idx * 10_000
        + replicate
    )


def run_replicate(task):

    (n, delta_label, delta, p_label, p_true,
     ratio_label, variance_ratio, replicate, seed) = task

    sigma2_true = float(np.sqrt(variance_ratio) * sigma1_true)
    sigma_avg = float(np.sqrt((sigma1_true**2 + sigma2_true**2) / 2.0))

    mu1_true = 0.0
    mu2_true = delta * sigma_avg

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
        "n": n,
        "delta_label": delta_label,
        "delta": delta,
        "p_label": p_label,
        "p_true": p_true,
        "ratio_label": ratio_label,
        "variance_ratio": variance_ratio,
        "mu1_true": mu1_true,
        "mu2_true": mu2_true,
        "sigma1_sq_true": sigma1_true**2,
        "sigma2_sq_true": sigma2_true**2,
        "replicate": replicate,
        "seed": seed,
    }

    rows = []

    row = dict(base_row)
    row["method"] = "MM"
    t0 = time.perf_counter()
    try:
        result = cohen_mod.cohen_fit_safe(x)
    except Exception as e:
        result = None
        row["error"] = f"{type(e).__name__}: {e}"
    row["time_sec"] = time.perf_counter() - t0
    _fill_result(row, result, p_true, mu1_true, mu2_true, sigma1_true, sigma2_true)
    rows.append(row)

    row = dict(base_row)
    row["method"] = "MLE"
    t0 = time.perf_counter()
    try:
        result = em_mod.em_fit_safe(x, n_init=em_n_init, seed=seed)
    except Exception as e:
        result = None
        row["error"] = f"{type(e).__name__}: {e}"
    row["time_sec"] = time.perf_counter() - t0
    _fill_result(row, result, p_true, mu1_true, mu2_true, sigma1_true, sigma2_true)
    rows.append(row)

    return rows


def _fill_result(row, result, p_true, mu1_true, mu2_true, sigma1_true, sigma2_true):
    if result is None:
        row["success"] = False
        for key in ["p_hat", "mu1_hat", "mu2_hat", "sigma1_sq_hat", "sigma2_sq_hat",
                    "p_error", "mu1_error", "mu2_error", "sigma1_sq_error", "sigma2_sq_error",
                    "p_abs_error", "mu1_abs_error", "mu2_abs_error",
                    "sigma1_sq_abs_error", "sigma2_sq_abs_error"]:
            row[key] = np.nan
        return

    row["success"] = True
    row["p_hat"] = result["p"]
    row["mu1_hat"] = result["mu1"]
    row["mu2_hat"] = result["mu2"]
    row["sigma1_sq_hat"] = result["sigma1_sq"]
    row["sigma2_sq_hat"] = result["sigma2_sq"]

    row["p_error"] = result["p"] - p_true
    row["mu1_error"] = result["mu1"] - mu1_true
    row["mu2_error"] = result["mu2"] - mu2_true
    row["sigma1_sq_error"] = result["sigma1_sq"] - sigma1_true**2
    row["sigma2_sq_error"] = result["sigma2_sq"] - sigma2_true**2

    row["p_abs_error"] = abs(row["p_error"])
    row["mu1_abs_error"] = abs(row["mu1_error"])
    row["mu2_abs_error"] = abs(row["mu2_error"])
    row["sigma1_sq_abs_error"] = abs(row["sigma1_sq_error"])
    row["sigma2_sq_abs_error"] = abs(row["sigma2_sq_error"])

def build_tasks():
    tasks = []
    for n_idx, n in enumerate(n_values):
        for delta_idx, (delta_label, delta) in enumerate(separations.items()):
            for p_idx, (p_label, p_true) in enumerate(p_values.items()):
                for ratio_idx, (ratio_label, variance_ratio) in enumerate(variance_ratios.items()):
                    for replicate in range(n_replicates):
                        seed = make_seed(n_idx, delta_idx, p_idx, ratio_idx, replicate)
                        tasks.append((
                            n, delta_label, delta, p_label, p_true,
                            ratio_label, variance_ratio, replicate, seed
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


if __name__ == "__main__":
    main()