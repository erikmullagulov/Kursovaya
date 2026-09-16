import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


 # log-likelihood
def mixture_log_likelihood(
    x,
    p,
    mu1,
    mu2,
    sigma1_sq,
    sigma2_sq
):
    sigma1 = np.sqrt(sigma1_sq)
    sigma2 = np.sqrt(sigma2_sq)

    log_f1 = (
        np.log(p) + norm.logpdf(x, mu1, sigma1)
    )

    log_f2 = (
        np.log(1.0 - p) + norm.logpdf(x, mu2, sigma2)
    )

    return np.sum(
        logsumexp(
            np.vstack([log_f1, log_f2]),
            axis=0
        )
    )


 # symmetric split around the sample mean
def em_initial_params_mean_split(x, rng, spread=1.0):

    mean = np.mean(x)
    std = np.std(x)

    p = 0.5

    mu1 = mean - spread * std
    mu2 = mean + spread * std

    sigma1_sq = std**2
    sigma2_sq = std**2

    return (
        p,
        mu1,
        mu2,
        sigma1_sq,
        sigma2_sq
    )


 # split at a random quantile
def em_initial_params_quantile_split(x, rng):

    x_sorted = np.sort(x)
    n = x.size

    cut_frac = rng.uniform(0.25, 0.75)
    cut = int(np.clip(round(cut_frac * n), 4, n - 4))

    left = x_sorted[:cut]
    right = x_sorted[cut:]

    p = left.size / n

    mu1 = np.mean(left)
    mu2 = np.mean(right)

    sigma1_sq = np.var(left) if left.size > 1 else np.var(x)
    sigma2_sq = np.var(right) if right.size > 1 else np.var(x)

    return (
        p,
        mu1,
        mu2,
        sigma1_sq,
        sigma2_sq
    )


 # random perturbation
def em_initial_params_perturbed(x, rng, jitter=0.75):

    mean = np.mean(x)
    std = np.std(x)

    mu1 = mean + rng.normal(scale=jitter * std)
    mu2 = mean + rng.normal(scale=jitter * std)

    if mu1 > mu2:
        mu1, mu2 = mu2, mu1

    p = rng.uniform(0.3, 0.7)

    sigma1_sq = std**2 * rng.uniform(0.5, 1.5)
    sigma2_sq = std**2 * rng.uniform(0.5, 1.5)

    return (
        p,
        mu1,
        mu2,
        sigma1_sq,
        sigma2_sq
    )


 # initialization strategies
init_strategies = [
    em_initial_params_mean_split,
    em_initial_params_quantile_split,
    em_initial_params_perturbed
]


 # EM method
def em_fit_single(
    x,
    p,
    mu1,
    mu2,
    sigma1_sq,
    sigma2_sq,
    tol=1e-6,
    max_iter=500,
    min_variance=1e-6,
    min_effective_weight=1e-8
):

    sigma1_sq = max(
        sigma1_sq,
        min_variance
    )

    sigma2_sq = max(
        sigma2_sq,
        min_variance
    )

    log_likelihood_old = -np.inf

    converged = False
    degenerate = False

     # EM iterations
    for iterations in range(1, max_iter + 1):

         # E-step
        sigma1 = np.sqrt(sigma1_sq)
        sigma2 = np.sqrt(sigma2_sq)

        log_prob_1 = (
            np.log(p) + norm.logpdf(
            x,
            mu1,
            sigma1)
        )

        log_prob_2 = (
            np.log(1.0 - p)
            + norm.logpdf(
                x,
                mu2,
                sigma2
            )
        )

        log_denom = logsumexp(
            np.vstack([
                log_prob_1,
                log_prob_2
            ]),
            axis=0
        )

        gamma = np.exp(
            log_prob_1 - log_denom
        )

         # M-step
        n1 = np.sum(gamma)
        n2 = np.sum(1.0 - gamma)

         # degenerate component
        if (
            n1 <= min_effective_weight * x.size
            or n2 <= min_effective_weight * x.size
        ):
            degenerate = True
            break

        p_new = n1 / x.size

        mu1_new = (
            np.sum(gamma * x)
            / n1
        )

        mu2_new = (
            np.sum((1.0 - gamma) * x)
            / n2
        )

        sigma1_sq_new = (
            np.sum(
                gamma * (x - mu1_new)**2
            )
            / n1
        )

        sigma2_sq_new = (
            np.sum(
                (1.0 - gamma) * (x - mu2_new)**2
            )
            / n2
        )

         # variance floor
        sigma1_sq_new = max(
            sigma1_sq_new,
            min_variance
        )

        sigma2_sq_new = max(
            sigma2_sq_new,
            min_variance
        )

         # new log-likelihood
        log_likelihood_new = (
            mixture_log_likelihood(
                x,
                p_new,
                mu1_new,
                mu2_new,
                sigma1_sq_new,
                sigma2_sq_new
            )
        )

         # update parameters
        p = p_new
        mu1 = mu1_new
        mu2 = mu2_new
        sigma1_sq = sigma1_sq_new
        sigma2_sq = sigma2_sq_new

         # convergence
        if abs(log_likelihood_new - log_likelihood_old) < tol:
            converged = True

            log_likelihood_old = (
                log_likelihood_new
            )
            break

        log_likelihood_old = (
            log_likelihood_new
        )

    if degenerate:
        return None

     # order components
    if mu1 > mu2:

        p = 1.0 - p

        mu1, mu2 = mu2, mu1

        sigma1_sq, sigma2_sq = (
            sigma2_sq,
            sigma1_sq
        )

    return {
        "p": float(p),
        "mu1": float(mu1),
        "mu2": float(mu2),
        "sigma1_sq": float(sigma1_sq),
        "sigma2_sq": float(sigma2_sq),
        "log_likelihood": float(
            log_likelihood_old
        ),
        "iterations": iterations,
        "converged": converged
    }


 # EM-method with multiple restarts
def em_fit(
    x,
    n_init=10,
    tol=1e-6,
    max_iter=500,
    min_variance=1e-6,
    seed=None
):

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        x = x.ravel()

    if x.size < 2:
        raise ValueError(
            "Need at least 2 observations"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Sample contains NaN or inf"
        )

    rng = np.random.default_rng(seed)

    results = []
    n_degenerate = 0

     # restarts
    for i in range(n_init):

        strategy = init_strategies[
            i % len(init_strategies)
        ]

        (
            p0,
            mu1_0,
            mu2_0,
            sigma1_sq_0,
            sigma2_sq_0
        ) = strategy(x, rng)

        run = em_fit_single(
            x,
            p0,
            mu1_0,
            mu2_0,
            sigma1_sq_0,
            sigma2_sq_0,
            tol=tol,
            max_iter=max_iter,
            min_variance=min_variance
        )

        if run is None:
            n_degenerate += 1
            continue

        results.append(run)

    if not results:
        raise RuntimeError(
            f"All {n_init} EM restarts collapsed onto a degenerate component"
        )

     # best restart
    best = max(
        results,
        key=lambda r: r["log_likelihood"]
    )

    best["n_restarts_tried"] = n_init
    best["n_restarts_degenerate"] = n_degenerate
    best["n_restarts_successful"] = len(results)

    best["log_likelihood_spread"] = (
        max(r["log_likelihood"] for r in results)
        - min(r["log_likelihood"] for r in results)
    )

    return best


def em_fit_safe(x, **kwargs):

    try:
        return em_fit(x, **kwargs)
    except Exception:
        return None


 # mixture generation
def create_mixture(
    n,
    p,
    mu1,
    mu2,
    sigma1,
    sigma2,
    seed=None
):

    rng = np.random.default_rng(seed)

    component = rng.random(n) < p

    x = np.where(
        component,
        rng.normal(mu1, sigma1, n),
        rng.normal(mu2, sigma2, n)
    )

    return x


if __name__ == "__main__":

    p_true = 0.5
    mu1_true = 0.0
    mu2_true = 3.0
    sigma1_true = 1.0
    sigma2_true = 1.0

    x = create_mixture(
        n=100,
        p=p_true,
        mu1=mu1_true,
        mu2=mu2_true,
        sigma1=sigma1_true,
        sigma2=sigma2_true,
        seed=31415
    )

    result = em_fit(x, n_init=10, seed=42)

    print("EM:")

    print(f"p        = {result['p']:.6f}")
    print(f"mu1      = {result['mu1']:.6f}")
    print(f"mu2      = {result['mu2']:.6f}")
    print(f"sigma1²  = {result['sigma1_sq']:.6f}")
    print(f"sigma2²  = {result['sigma2_sq']:.6f}")

    print("\nПроверка:")

    print(
        f"log-likelihood = {result['log_likelihood']:.6f}"
    )

    print(
        f"iterations     = "
        f"{result['iterations']}"
    )

    print(
        f"converged      = "
        f"{result['converged']}"
    )

    print(
        f"restarts tried       = "
        f"{result['n_restarts_tried']}"
    )

    print(
        f"restarts degenerate  = "
        f"{result['n_restarts_degenerate']}"
    )

    print(
        f"restarts successful  = "
        f"{result['n_restarts_successful']}"
    )

    print(
        f"log-like spread       = "
        f"{result['log_likelihood_spread']:.6f}"
    )