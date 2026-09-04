"""
Perceptual noise model: what a player can actually know at decision time.

Our geometry knows where the pitch crossed to within a fraction of an inch. A
player judging a moving pitch does not. Conflating the two makes any "runs left
on the table" number an upper bound that mixes a decision gap with an
information gap.

Sigma is estimated WITHOUT reference to challenge volume or success rate, both
of which are the things we are trying to explain. Fitting sigma to those would
be circular -- one parameter fitted to the two observables that jointly
constitute the measurement, forcing the gap to zero by construction.

Instead sigma is identified from the SHAPE of the distribution of true pitch
locations among the calls players actually chose to challenge. Under perfect
information nobody challenges a pitch sitting well inside the zone. Every such
hopeless challenge in the data is direct evidence of perceptual noise, and how
far in they go pins down how much.

Sign convention throughout: d = signed ball-edge distance in feet, POSITIVE
means the pitch is in the zone (a correct strike call). A batter challenging a
called strike wins if d < 0; a catcher challenging a called ball wins if d > 0.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def challenge_probability(d, sigma, cutoff, side):
    """
    P(player challenges | true distance d). The player observes o = d + noise
    and challenges when the observation crosses their cutoff.

    batting side: challenges called strikes, wants d < 0, fires when o < cutoff
    fielding side: challenges called balls, wants d > 0, fires when o > cutoff
    """
    if side == "batting":
        return norm.cdf((cutoff - d) / sigma)
    return norm.cdf((d - cutoff) / sigma)


def negative_log_likelihood(params, d_base, d_challenged, side):
    """
    Likelihood of the observed challenged-pitch locations under (sigma, cutoff).

    The density of d among challenges is  f(d) * P(challenge|d) / Z, where f is
    the base distribution of all called pitches of this type. f(d_i) does not
    depend on the parameters and drops out; Z is estimated as the mean of
    P(challenge|d) over the base sample.
    """
    log_sigma, cutoff = params
    sigma = np.exp(log_sigma)
    p_chal = challenge_probability(d_challenged, sigma, cutoff, side)
    Z = challenge_probability(d_base, sigma, cutoff, side).mean()
    if Z <= 0 or not np.isfinite(Z):
        return 1e12
    ll = np.log(np.clip(p_chal, 1e-300, None)).sum() - len(d_challenged) * np.log(Z)
    return -ll if np.isfinite(ll) else 1e12


def fit_sigma(d_base, d_challenged, side, sigma_init_ft=1.5 / 12, cutoff_init=0.0):
    """MLE for (sigma, cutoff). Returns sigma in feet and the cutoff in feet."""
    res = minimize(
        negative_log_likelihood,
        x0=[np.log(sigma_init_ft), cutoff_init],
        args=(d_base, d_challenged, side),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
    )
    return np.exp(res.x[0]), res.x[1], -res.fun


def likelihood_surface(d_base, d_challenged, side, sigmas_ft, cutoffs_ft):
    """Grid of log-likelihood for identifiability inspection (ridge vs peak)."""
    out = np.empty((len(sigmas_ft), len(cutoffs_ft)))
    for i, s in enumerate(sigmas_ft):
        for j, c in enumerate(cutoffs_ft):
            out[i, j] = -negative_log_likelihood([np.log(s), c], d_base, d_challenged, side)
    return out


def build_posterior_lookup(d_base, sigma, side, n_obs_grid=1500, n_d_grid=400):
    """
    P(challenge succeeds | observation o), using the empirical distribution of
    true locations as the prior. Returns (o_grid, p_grid) for interpolation --
    the posterior depends only on o, so it can be tabulated once.
    """
    lo, hi = np.percentile(d_base, [0.05, 99.95])
    pad = 6 * sigma
    d_grid = np.linspace(lo - pad, hi + pad, n_d_grid)
    prior, _ = np.histogram(d_base, bins=np.append(
        d_grid - (d_grid[1] - d_grid[0]) / 2, d_grid[-1] + (d_grid[1] - d_grid[0]) / 2))
    prior = prior.astype(float)
    prior /= prior.sum()

    o_grid = np.linspace(d_grid[0] - pad, d_grid[-1] + pad, n_obs_grid)
    # likelihood[i, j] = N(o_i ; d_j, sigma)
    like = norm.pdf((o_grid[:, None] - d_grid[None, :]) / sigma) / sigma
    joint = like * prior[None, :]
    denom = joint.sum(axis=1)
    win_mask = (d_grid < 0) if side == "batting" else (d_grid > 0)
    num = joint[:, win_mask].sum(axis=1)
    p_win = np.where(denom > 0, num / np.maximum(denom, 1e-300), 0.0)
    return o_grid, p_win


def posterior_from_observation(o, o_grid, p_grid):
    return np.interp(o, o_grid, p_grid)
