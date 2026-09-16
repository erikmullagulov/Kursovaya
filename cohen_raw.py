import numpy as np

from scipy.optimize import root

 # mixture generation
def mixture(n, p, mu1, mu2, sigma1, sigma2, seed=None):
    rng = np.random.default_rng(seed)

    component = rng.random(n) < p

    x = np.where(
        component,
        rng.normal(mu1, sigma1, n),
        rng.normal(mu2, sigma2, n)
    )

    return x

x = mixture(100, 0.5, 0, 3, 1, 1, seed=31415)
print(x[:6])

 # sample moments calculation
def sample_moments(x):
    moments = {}

    for k in range(1, 6):
        moments[k] = np.mean(x ** k)

    return moments

for key, value in sample_moments(x).items():
    print(f"m_{key} =  {value}")

moments = sample_moments(x)

m1 = moments[1]
m2 = moments[2]
m3 = moments[3]
m4 = moments[4]
m5 = moments[5]

 # difference between theoretical moments and sample moments assuming equal variances
def eq_v_th_moments(params):

    p, mu1, mu2, sigma_sq = params

    m1_th = (
        p * mu1 + (1 - p) * mu2
    )

    m2_th = (
        p * (mu1**2 + sigma_sq) + (1 - p) * (mu2**2 + sigma_sq)
    )

    m3_th = (
        p * (mu1**3 + 3 * mu1 * sigma_sq) + (1 - p) * (mu2**3 + 3 * mu2 * sigma_sq)
    )

    m4_th = (
        p * (
            mu1**4  + 6 * mu1**2 * sigma_sq + 3 * sigma_sq**2
        )
        + (1 - p) * (
            mu2**4+ 6 * mu2**2 * sigma_sq + 3 * sigma_sq**2
        )
    )

    return np.array([
        m1_th - m1,
        m2_th - m2,
        m3_th - m3,
        m4_th - m4
    ])

sample_mean = np.mean(x)
sample_std = np.std(x)

initial_guess_eq_v = np.array([
    0.5,
    sample_mean - sample_std,
    sample_mean + sample_std,
    sample_std**2
])

result_eq_v = root(
    eq_v_th_moments,
    initial_guess_eq_v
)

print("\nEqual variances initial estimates:")

p_eq, mu1_eq, mu2_eq, sigma_sq_eq = result_eq_v.x

sigma_eq = np.sqrt(sigma_sq_eq)

print(f"p       = {p_eq:.6f}")
print(f"mu1     = {mu1_eq:.6f}")
print(f"mu2     = {mu2_eq:.6f}")
print(f"sigma   = {sigma_eq:.6f}")

 # initial values for the general case
p = p_eq
mu2 = mu2_eq

mu1 = (
    m1 - (1 - p) * mu2
) / p

A = (
    m2 - p * mu1**2 - (1 - p) * mu2**2
)

B = (
    m3 - p * mu1**3 - (1 - p) * mu2**3
) / 3

sigma2_sq = (
    B - mu1 * A
) / (
    (1 - p) * (mu2 - mu1)
)

sigma1_sq = (
    A - (1 - p) * sigma2_sq
) / p

print("\nInitial estimates for the general case:")

print(f"p        = {p:.6f}")
print(f"mu1      = {mu1:.6f}")
print(f"mu2      = {mu2:.6f}")
print(f"sigma1²  = {sigma1_sq:.6f}")
print(f"sigma2²  = {sigma2_sq:.6f}")

 # central moments of the sample
v2 = np.mean((x - sample_mean) ** 2)
v3 = np.mean((x - sample_mean) ** 3)
v4 = np.mean((x - sample_mean) ** 4)
v5 = np.mean((x - sample_mean) ** 5)

 # fourth cumulant
k4 = v4 - 3 * v2**2

print("\nCentral moments:")
print(f"v2 = {v2:.6f}")
print(f"v3 = {v3:.6f}")
print(f"v4 = {v4:.6f}")
print(f"v5 = {v5:.6f}")

print("\nFourth cumulant:")
print(f"k4 = {k4:.6f}")

 # check special cases
print("\nChecking Cohen's special cases:")

if abs(v3) < 1e-10 and k4 < 0:
    print("Symmetric case: p = 0.5 and sigma1 = sigma2")
elif abs(v3) < 1e-10 and k4 > 0:
    print("Symmetric case: mu1 = mu2 and sigma1 != sigma2")
elif abs(v3) < 1e-10 and abs(k4) < 1e-10:
    print("Single normal distribution")
else:
    print("General asymmetric case")

 # initial approximation to r from the equal variances case
m1_centered_eq = mu1_eq - sample_mean
m2_centered_eq = mu2_eq - sample_mean

r = m1_centered_eq + m2_centered_eq

print("\nInitial approximation for the general case:")
print(f"m1_centered = {m1_centered_eq:.6f}")
print(f"m2_centered = {m2_centered_eq:.6f}")
print(f"r0          = {r:.6f}")

 # Cohen's general cubic equation (29)
coefficients = [
    6,
    -2 * r**2,
    3 * k4 - 4 * r * v3,
    v3**2
]

roots_v = np.roots(coefficients)

print("\nRoots of Cohen's cubic equation:")
for z in roots_v:
    print(z)

 # need negative real root
negative_real_roots = [
    z.real
    for z in roots_v
    if abs(z.imag) < 1e-10 and z.real < 0
]

if len(negative_real_roots) != 1:
    raise ValueError(
        f"Expected exactly one negative real root, got: {negative_real_roots}"
    )

v = negative_real_roots[0]

print("\nSelected negative real root:")
print(f"v = {v:.6f}")

 # finding centered component means m1 and m2 from r = m1 + m2 and v = m1 * m2
discriminant = r**2 - 4 * v

if discriminant < 0:
    raise ValueError("Negative discriminant: m1 and m2 are not real.")

m1_centered = (
    r - np.sqrt(discriminant)
) / 2

m2_centered = (
    r + np.sqrt(discriminant)
) / 2

print("\nCentered component means:")
print(f"m1 = {m1_centered:.6f}")
print(f"m2 = {m2_centered:.6f}")

print("\nChecks:")
print(f"r = m1 + m2 = {m1_centered + m2_centered:.6f}")
print(f"v = m1 * m2 = {m1_centered * m2_centered:.6f}")

 # estimating mixing proportion p from the centered component means
p_general = (
    m2_centered
    / (m2_centered - m1_centered)
)

print("\nMixing proportion:")
print(f"p = {p_general:.6f}")

 # checking that p is viable
if not (0 < p_general < 1):
    raise ValueError("Estimated p lies outside of the interval (0, 1).")

 # estimate the component variances from second and third central moments
A = (
    v2 - p_general * m1_centered**2 - (1 - p_general) * m2_centered**2
)

B = (
    v3 - p_general * m1_centered**3 - (1 - p_general) * m2_centered**3
) / 3

sigma2_sq_general = (
    B - m1_centered * A
) / (
    (1 - p_general) * (m2_centered - m1_centered)
)

sigma1_sq_general = (
    A - (1 - p_general) * sigma2_sq_general
) / p_general

print("\nComponent variances:")
print(f"sigma1² = {sigma1_sq_general:.6f}")
print(f"sigma2² = {sigma2_sq_general:.6f}")

 # check that variances are positive
if sigma1_sq_general <= 0 or sigma2_sq_general <= 0:
    raise ValueError("Estimated variance is not positive.")

 # theoretical fifth central moment for the current general case estimates
mu5_fit = (
        p_general * (
        m1_centered ** 5
        + 10 * m1_centered ** 3 * sigma1_sq_general
        + 15 * m1_centered * sigma1_sq_general ** 2
)
        + (1 - p_general) * (
                m2_centered ** 5
                + 10 * m2_centered ** 3 * sigma2_sq_general
                + 15 * m2_centered * sigma2_sq_general ** 2
        )
)

print("\nFitted fifth central moment:")
print(f"mu5_fit = {mu5_fit:.6f}")

print("\nComparison with sample moment:")
print(f"v5         = {v5:.6f}")
print(f"difference = {mu5_fit - v5:.6f}")

def cohen_trial(r):
     # (29)
    coefficients = [
        6,
        -2 * r**2,
        3 * k4 - 4 * r * v3,
        v3**2
    ]

    roots_v = np.roots(coefficients)

    negative_real_roots = [
        z.real
        for z in roots_v
        if abs(z.imag) < 1e-10 and z.real < 0
    ]

    if len(negative_real_roots) != 1:
        raise ValueError(
            f"Expected one negative real root, got: {negative_real_roots}"
        )

    v = negative_real_roots[0]

     # centered component means
    discriminant = r**2 - 4 * v

    if discriminant < 0:
        raise ValueError("Negative discriminant.")

    m1_c = (r - np.sqrt(discriminant)) / 2
    m2_c = (r + np.sqrt(discriminant)) / 2

     # mixing proportion
    p = m2_c / (m2_c - m1_c)

    if not (0 < p < 1):
        raise ValueError("Invalid mixing proportion.")

     # component variances
    A = (
        v2 - p * m1_c**2 - (1 - p) * m2_c**2
    )

    B = (
        v3 - p * m1_c**3 - (1 - p) * m2_c**3
    ) / 3

    sigma2_sq = (
        B - m1_c * A
    ) / (
        (1 - p) * (m2_c - m1_c)
    )

    sigma1_sq = (
        A - (1 - p) * sigma2_sq
    ) / p

    if sigma1_sq <= 0 or sigma2_sq <= 0:
        raise ValueError("Non-positive variance.")

     # fitted fifth central moment
    mu5 = (
        p * (
            m1_c**5 + 10 * m1_c**3 * sigma1_sq + 15 * m1_c * sigma1_sq**2
        )
        + (1 - p) * (
            m2_c**5 + 10 * m2_c**3 * sigma2_sq + 15 * m2_c * sigma2_sq**2
        )
    )

    return {
        "r": r,
        "v": v,
        "m1": m1_c,
        "m2": m2_c,
        "p": p,
        "sigma1_sq": sigma1_sq,
        "sigma2_sq": sigma2_sq,
        "mu5": mu5
    }

 # checking cohen_trial for r0
trial = cohen_trial(r)

print("\nCohen trial for r0:")
print(f"r         = {trial['r']:.6f}")
print(f"v         = {trial['v']:.6f}")
print(f"m1        = {trial['m1']:.6f}")
print(f"m2        = {trial['m2']:.6f}")
print(f"p         = {trial['p']:.6f}")
print(f"sigma1²   = {trial['sigma1_sq']:.6f}")
print(f"sigma2²   = {trial['sigma2_sq']:.6f}")
print(f"mu5       = {trial['mu5']:.6f}")

 # checking second trial value of r
r_test = -0.5
trial_2 = cohen_trial(r_test)

print("\nCohen trial for second r:")
print(f"r         = {trial_2['r']:.6f}")
print(f"v         = {trial_2['v']:.6f}")
print(f"m1        = {trial_2['m1']:.6f}")
print(f"m2        = {trial_2['m2']:.6f}")
print(f"p         = {trial_2['p']:.6f}")
print(f"sigma1²   = {trial_2['sigma1_sq']:.6f}")
print(f"sigma2²   = {trial_2['sigma2_sq']:.6f}")
print(f"mu5       = {trial_2['mu5']:.6f}")

 # checking third trial value of r
r_test = 0.5
trial_3 = cohen_trial(r_test)

print("\nCohen trial for 3rd r:")
print(f"r         = {trial_3['r']:.6f}")
print(f"v         = {trial_3['v']:.6f}")
print(f"m1        = {trial_3['m1']:.6f}")
print(f"m2        = {trial_3['m2']:.6f}")
print(f"p         = {trial_3['p']:.6f}")
print(f"sigma1²   = {trial_3['sigma1_sq']:.6f}")
print(f"sigma2²   = {trial_3['sigma2_sq']:.6f}")
print(f"mu5       = {trial_3['mu5']:.6f}")

 # checking fourth trial value of r
r_test = 1
trial_4 = cohen_trial(r_test)

print("\nCohen trial for 4th r:")
print(f"r         = {trial_4['r']:.6f}")
print(f"v         = {trial_4['v']:.6f}")
print(f"m1        = {trial_4['m1']:.6f}")
print(f"m2        = {trial_4['m2']:.6f}")
print(f"p         = {trial_4['p']:.6f}")
print(f"sigma1²   = {trial_4['sigma1_sq']:.6f}")
print(f"sigma2²   = {trial_4['sigma2_sq']:.6f}")
print(f"mu5       = {trial_4['mu5']:.6f}")

 # checking fifth trial value of r
r_test = 1.2
trial_5 = cohen_trial(r_test)

print("\nCohen trial for 5th r:")
print(f"r         = {trial_5['r']:.6f}")
print(f"v         = {trial_5['v']:.6f}")
print(f"m1        = {trial_5['m1']:.6f}")
print(f"m2        = {trial_5['m2']:.6f}")
print(f"p         = {trial_5['p']:.6f}")
print(f"sigma1²   = {trial_5['sigma1_sq']:.6f}")
print(f"sigma2²   = {trial_5['sigma2_sq']:.6f}")
print(f"mu5       = {trial_5['mu5']:.6f}")

 # first linear interpolation for r*
r1 = 1.0
mu5_1 = trial_4["mu5"]

r2 = 1.2
mu5_2 = trial_5["mu5"]

r_star_1 = r1 + (
    (v5 - mu5_1) * (r2 - r1) / (mu5_2 - mu5_1)
)

print("\nFirst interpolation for r*:")
print(f"r1       = {r1:.6f}")
print(f"mu5(r1)  = {mu5_1:.6f}")
print(f"r2       = {r2:.6f}")
print(f"mu5(r2)  = {mu5_2:.6f}")
print(f"r*       = {r_star_1:.6f}")

 # checking the interpolated value of r*

trial_star = cohen_trial(r_star_1)

print("\nCohen estimates at interpolated r*:")
print(f"r         = {trial_star['r']:.6f}")
print(f"v         = {trial_star['v']:.6f}")
print(f"m1        = {trial_star['m1']:.6f}")
print(f"m2        = {trial_star['m2']:.6f}")
print(f"p         = {trial_star['p']:.6f}")
print(f"sigma1²   = {trial_star['sigma1_sq']:.6f}")
print(f"sigma2²   = {trial_star['sigma2_sq']:.6f}")
print(f"mu5       = {trial_star['mu5']:.6f}")

print("\nComparison:")
print(f"sample v5  = {v5:.6f}")
print(f"difference = {trial_star['mu5'] - v5:.6f}")

 # second linear interpolation for r*

r1 = r_star_1
mu5_1 = trial_star["mu5"]

r2 = 1.2
mu5_2 = trial_5["mu5"]

r_star_2 = r1 + (
    (v5 - mu5_1) * (r2 - r1) / (mu5_2 - mu5_1)
)

print("\nSecond interpolation for r*:")
print(f"r1       = {r1:.6f}")
print(f"mu5(r1)  = {mu5_1:.6f}")
print(f"r2       = {r2:.6f}")
print(f"mu5(r2)  = {mu5_2:.6f}")
print(f"r*       = {r_star_2:.6f}")

 # final Cohen estimates

cohen_final = cohen_trial(r_star_2)

mu1_cohen = sample_mean + cohen_final["m1"]
mu2_cohen = sample_mean + cohen_final["m2"]

print("\nFinal Cohen estimates:")
print(f"r         = {cohen_final['r']:.6f}")
print(f"p         = {cohen_final['p']:.6f}")
print(f"mu1       = {mu1_cohen:.6f}")
print(f"mu2       = {mu2_cohen:.6f}")
print(f"sigma1²   = {cohen_final['sigma1_sq']:.6f}")
print(f"sigma2²   = {cohen_final['sigma2_sq']:.6f}")
print(f"mu5       = {cohen_final['mu5']:.6f}")

print("\nComparison with sample fifth moment:")
print(f"sample v5 = {v5:.6f}")
print(f"difference = {cohen_final['mu5'] - v5:.6f}")