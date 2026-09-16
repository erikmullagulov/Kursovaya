import time
import traceback

import numpy as np
import pandas as pd

import cohen_director_cut as cohen_mod
import mle_with_no_L as em_mod


n_values = [20, 50, 100, 300, 1000]

 # separation delta = |mu1 - mu2| / sigma, sigma1 = sigma2 = 1, mu1 = 0
separations = {
    "weak": 0.5,
    "medium": 1.5,
    "strong": 3.0
}

p_values = {
    "balanced": 0.5,
    "unbalanced": 0.2
}

n_replicates = 500

sigma1_true = 1.0
sigma2_true = 1.0

em_n_init = 10

 # output files
results_path = "simulation_results.csv"
checkpoint_every = 50


 # unique seed per (n, delta, p, replicate) cell
def make_seed(n_idx, delta_idx, p_idx, replicate):
    return (
        n_idx * 10_000_000
        + delta_idx * 1_000_000
        + p_idx * 100_000
        + replicate
    )


 # run a single replicate for both methods
def run_replicate(n, delta_label, delta, p_label, p_true, replicate, seed):

    mu1_true = 0.0
    mu2_true = delta * sigma1_true

    rows = []

    x = cohen_mod.create_mixture(
        n=n,
        p=p_true,
        mu1=mu1_true,
        mu2=mu2_true,
        sigma1=sigma1_true,
        sigma2=sigma2_true,
        seed=seed
    )

    base_row = {
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
        "seed": seed
    }

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

    if result is None:
        row["success"] = False
        row["p_hat"] = np.nan
        row["mu1_hat"] = np.nan
        row["mu2_hat"] = np.nan
        row["sigma1_sq_hat"] = np.nan
        row["sigma2_sq_hat"] = np.nan
        row["p_error"] = np.nan
        row["mu1_error"] = np.nan
        row["mu2_error"] = np.nan
        row["sigma1_sq_error"] = np.nan
        row["sigma2_sq_error"] = np.nan
        row["p_abs_error"] = np.nan
        row["mu1_abs_error"] = np.nan
        row["mu2_abs_error"] = np.nan
        row["sigma1_sq_abs_error"] = np.nan
        row["sigma2_sq_abs_error"] = np.nan
    else:
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

    if result is None:
        row["success"] = False
        row["p_hat"] = np.nan
        row["mu1_hat"] = np.nan
        row["mu2_hat"] = np.nan
        row["sigma1_sq_hat"] = np.nan
        row["sigma2_sq_hat"] = np.nan
        row["p_error"] = np.nan
        row["mu1_error"] = np.nan
        row["mu2_error"] = np.nan
        row["sigma1_sq_error"] = np.nan
        row["sigma2_sq_error"] = np.nan
        row["p_abs_error"] = np.nan
        row["mu1_abs_error"] = np.nan
        row["mu2_abs_error"] = np.nan
        row["sigma1_sq_abs_error"] = np.nan
        row["sigma2_sq_abs_error"] = np.nan
    else:
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

    rows.append(row)

    return rows


 # main loop

def main():

    all_rows = []

    n_cells = (
        len(n_values)
        * len(separations)
        * len(p_values)
    )

    cell_idx = 0
    start_time = time.time()

    for n_idx, n in enumerate(n_values):
        for delta_idx, (delta_label, delta) in enumerate(separations.items()):
            for p_idx, (p_label, p_true) in enumerate(p_values.items()):

                cell_idx += 1

                print(
                    f"[{cell_idx}/{n_cells}] "
                    f"n={n}, delta={delta_label} ({delta}), "
                    f"p={p_label} ({p_true})"
                )

                for replicate in range(n_replicates):

                    seed = make_seed(
                        n_idx,
                        delta_idx,
                        p_idx,
                        replicate
                    )

                    try:
                        rows = run_replicate(
                            n,
                            delta_label,
                            delta,
                            p_label,
                            p_true,
                            replicate,
                            seed
                        )
                    except Exception:
                        print(
                            f"UNEXPECTED FAILURE at "
                            f"n={n}, delta={delta_label}, "
                            f"p={p_label}, replicate={replicate}"
                        )
                        traceback.print_exc()
                        continue

                    all_rows.extend(rows)

                    if (replicate + 1) % checkpoint_every == 0:
                        pd.DataFrame(all_rows).to_csv(
                            results_path,
                            index=False
                        )

                 # checkpoint after every full cell
                pd.DataFrame(all_rows).to_csv(
                    results_path,
                    index=False
                )

    elapsed = time.time() - start_time

    df = pd.DataFrame(all_rows)
    df.to_csv(results_path, index=False)

    print(f"\nDone. {len(df)} rows written to {results_path}")
    print(f"Total time: {elapsed / 60:.1f} min")

     # success rate summary

    success_summary = (
        df.groupby(["method", "n", "delta_label", "p_label"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate"})
    )

    print("\nSuccess rate by cell:")
    print(success_summary.to_string(index=False))

     # average absolute errors (only successful runs)
    error_cols = [
        "p_abs_error",
        "mu1_abs_error",
        "mu2_abs_error",
        "sigma1_sq_abs_error",
        "sigma2_sq_abs_error"
    ]

    error_summary = (
        df[df["success"]]
        .groupby(["method", "n", "delta_label", "p_label"])[error_cols]
        .mean()
        .reset_index()
    )

    print("\nMean absolute errors (successful runs only):")
    print(error_summary.to_string(index=False))


if __name__ == "__main__":
    main()