import numpy as np


 # sample moments
def sample_central_moments(x):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        x = x.ravel()

    if x.size < 5:
        raise ValueError("For Cohen at least 5 observations needed")

    if not np.all(np.isfinite(x)):
        raise ValueError("Sample contains NaN or inf")

    mean = np.mean(x)
    z = x - mean

    v2 = np.mean(z**2)
    v3 = np.mean(z**3)
    v4 = np.mean(z**4)
    v5 = np.mean(z**5)

    return mean, v2, v3, v4, v5


 # negative real roots
def _negative_real_roots(coefficients, imag_tol=1e-9):
    roots = np.roots(coefficients)

    scale = max(
        1.0,
        np.max(np.abs(roots))
    )

    negative_roots = [
        z.real
        for z in roots
        if (
            abs(z.imag) <= imag_tol * scale
            and z.real < 0
        )
    ]

    return sorted(negative_roots)


 # initial r
def cohen_initial_r(v3, k4):

     # equal variance case
    roots = _negative_real_roots(
        [
            2.0,
            0.0,
            k4,
            v3**2
        ]
    )

    if len(roots) != 1:
        raise ValueError(
            "Could not find unique negative root"
        )

    v = roots[0]

    if np.isclose(v, 0.0, atol=1e-14):
        raise ValueError("Root v is too small")

     # initial r
    r0 = -v3 / v

    return float(r0)


 # trial for given r
def cohen_trial(r, v2, v3, k4):

     # cubic equation for v
    coefficients = [
        6.0,
        -2.0 * r**2,
        3.0 * k4 - 4.0 * r * v3,
        v3**2
    ]

    roots = _negative_real_roots(coefficients)

    if len(roots) != 1:
        raise ValueError(
            "Could not find unique negative root v"
        )

    v = roots[0]

     # centered means
    discriminant = r**2 - 4.0 * v
    if discriminant <= 0:
        discriminant = max(discriminant, 1e-12)  # avoid zero/negative
    delta = np.sqrt(discriminant)

    m1 = (r - delta) / 2.0
    m2 = (r + delta) / 2.0

     # mixing probability
    p = m2 / (m2 - m1)

    if not (0 < p < 1):
        raise ValueError("Invalid p")

     # variance equations
    A = (
        v2
        - p * m1**2
        - (1 - p) * m2**2
    )

    B = (
        v3
        - p * m1**3
        - (1 - p) * m2**3
    ) / 3.0

    sigma2_sq = (
        B - m1 * A
    ) / (
        (1 - p) * (m2 - m1)
    )

    sigma1_sq = (
        A - (1 - p) * sigma2_sq
    ) / p

    if sigma1_sq <= 0:
        raise ValueError("Invalid sigma1^2")

    if sigma2_sq <= 0:
        raise ValueError("Invalid sigma2^2")

     # fifth moment
    mu5 = (
        p * (
            m1**5
            + 10 * m1**3 * sigma1_sq
            + 15 * m1 * sigma1_sq**2
        )
        +
        (1 - p) * (
            m2**5
            + 10 * m2**3 * sigma2_sq
            + 15 * m2 * sigma2_sq**2
        )
    )

    return {
        "r": float(r),
        "v": float(v),
        "m1": float(m1),
        "m2": float(m2),
        "p": float(p),
        "sigma1_sq": float(sigma1_sq),
        "sigma2_sq": float(sigma2_sq),
        "mu5": float(mu5)
    }


 # find interval for r
def find_cohen_interval(
    r0,
    v2,
    v3,
    k4,
    v5,
    step_scale=0.25,
    growth=1.5,
    max_radius=20.0,
    max_steps=200
):

    scale = max(np.sqrt(v2), 1e-12)
    base_step = step_scale * scale

    def f(r):
        result = cohen_trial(r, v2, v3, k4)
        return result["mu5"] - v5

     # search in both directions
    for direction in (-1.0, 1.0):

        previous_r = r0

        try:
            previous_f = f(previous_r)
        except ValueError:
            previous_f = None

        step = base_step

        for _ in range(max_steps):

            current_r = r0 + direction * step

            try:
                current_f = f(current_r)
            except ValueError:
                step *= growth

                if step > max_radius * scale:
                    break

                previous_r = current_r
                previous_f = None

                continue

             # sign change
            if (
                previous_f is not None
                and previous_f * current_f <= 0
            ):
                return previous_r, current_r

            previous_r = current_r
            previous_f = current_f

            step *= growth

            if step > max_radius * scale:
                break

    raise ValueError(
        "Could not find valid bracket for r"
    )


 # Cohen method
def cohen_fit(
    x,
    tol=1e-8,
    max_iter=100,
    bracket_step=0.25
):

     # sample moments
    mean, v2, v3, v4, v5 = (
        sample_central_moments(x)
    )

    if v2 <= 0:
        raise ValueError("Sample variance is zero")

     # fourth cumulant
    k4 = v4 - 3 * v2**2

     # initial r
    r0 = cohen_initial_r(v3, k4)

     # find interval
    r_left, r_right = find_cohen_interval(
        r0,
        v2,
        v3,
        k4,
        v5,
        step_scale=bracket_step
    )

    left = cohen_trial(
        r_left,
        v2,
        v3,
        k4
    )

    right = cohen_trial(
        r_right,
        v2,
        v3,
        k4
    )

    f_left = left["mu5"] - v5
    f_right = right["mu5"] - v5

    current = None
    converged = False
    iterations = 0

     # interpolation
    for iterations in range(1, max_iter + 1):

        denominator = (
            right["mu5"]
            - left["mu5"]
        )

        if np.isclose(
            denominator,
            0.0,
            atol=1e-14
        ):
            raise RuntimeError(
                "Degenerate interpolation"
            )

        r = (
            r_left
            + (v5 - left["mu5"])
            * (r_right - r_left)
            / denominator
        )

         # new trial
        try:
            current = cohen_trial(
                r,
                v2,
                v3,
                k4
            )
        except ValueError:

             # fallback to midpoint
            r = 0.5 * (r_left + r_right)

            current = cohen_trial(
                r,
                v2,
                v3,
                k4
            )

         # fifth moment error
        f_current = current["mu5"] - v5

         # convergence
        if (
            abs(f_current) < tol
            or
            abs(r_right - r_left)
            < tol * max(1.0, abs(r))
        ):
            converged = True
            break

         # update bracket
        if f_left * f_current <= 0:
            r_right = r
            right = current
            f_right = f_current
        else:
            r_left = r
            left = current
            f_left = f_current

    if not converged:
        raise RuntimeError(
            f"Cohen did not converge in {max_iter} iterations"
        )

     # original means
    mu1 = mean + current["m1"]
    mu2 = mean + current["m2"]

     # ensure ordering
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        p = 1.0 - current["p"]
        sigma1_sq, sigma2_sq = current["sigma2_sq"], current["sigma1_sq"]
    else:
        p = current["p"]
        sigma1_sq = current["sigma1_sq"]
        sigma2_sq = current["sigma2_sq"]

    if not (0 < p < 1):
        raise RuntimeError("Final p out of (0,1)")

    return {
        "p": p,
        "mu1": mu1,
        "mu2": mu2,
        "sigma1_sq": sigma1_sq,
        "sigma2_sq": sigma2_sq,
        "r": current["r"],
        "v": current["v"],
        "sample_mean": mean,
        "v2": v2,
        "v3": v3,
        "v4": v4,
        "v5": v5,
        "k4": k4,
        "mu5_fitted": current["mu5"],
        "moment5_error": current["mu5"] - v5,
        "iterations": iterations,
        "converged": converged
    }


def cohen_fit_safe(x, **kwargs):

    try:
        return cohen_fit(x, **kwargs)
    except (ValueError, RuntimeError) as e:
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
    mu1_true = 0
    mu2_true = 3
    sigma1_true = 1
    sigma2_true = 1

    x = create_mixture(
        n=100,
        p=p_true,
        mu1=mu1_true,
        mu2=mu2_true,
        sigma1=sigma1_true,
        sigma2=sigma2_true,
        seed=31415
    )

    result = cohen_fit(x)

    print("Cohen:")
    print(f"p        = {result['p']:.6f}")
    print(f"mu1      = {result['mu1']:.6f}")
    print(f"mu2      = {result['mu2']:.6f}")
    print(f"sigma1^2 = {result['sigma1_sq']:.6f}")
    print(f"sigma2^2 = {result['sigma2_sq']:.6f}")

    print("\nDiagnostics:")
    print(f"r          = {result['r']:.6f}")
    print(f"v          = {result['v']:.6f}")
    print(f"mu5        = {result['mu5_fitted']:.12f}")
    print(f"v5         = {result['v5']:.12f}")
    print(f"error      = {result['moment5_error']:.3e}")
    print(f"iterations = {result['iterations']}")
    print(f"converged  = {result['converged']}")