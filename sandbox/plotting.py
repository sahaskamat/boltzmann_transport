"""Plotting helpers that accept theoretical or experimental curve data."""

import numpy as np

from .calculations import CurveResult
from .data import ExperimentalData


def plot_curve(ax, theta, rho, *, label=None, normalize=False, **plot_kwargs):
    """Plot one ``theta``/``rho`` curve on an existing Matplotlib axis.

    The function is intentionally unaware of where the data came from. The
    caller can use it for model output, experimental data, or both.
    """
    theta = np.asarray(theta)
    rho = np.asarray(rho)
    if theta.shape != rho.shape:
        raise ValueError("theta and rho must have the same shape")
    if normalize:
        rho = rho / rho[np.argmin(theta**2)]
    if label is not None:
        plot_kwargs["label"] = label
    return ax.plot(theta, rho, **plot_kwargs)


def plot_curves(
    theoretical_results: list[CurveResult],
    experimental_data: list[ExperimentalData] | None = None,
    show: bool = True,
):
    """Plot theoretical results and optional experimental curves together."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for result in theoretical_results:
        plot_curve(
            axes[0], result.theta, result.rhozz,
            label=f"Model, {result.config.label}", marker="o", ms=2,
        )
        plot_curve(
            axes[1], result.theta, result.rhozz,
            label=f"Model, {result.config.label}", normalize=True,
            marker="o", ms=2,
        )

    for data in experimental_data or []:
        plot_curve(
            axes[0], data.theta, data.rho,
            label=data.label, marker="o", ms=2,
        )
        plot_curve(
            axes[1], data.theta, data.rho,
            label=data.label, normalize=True, marker="o", ms=2,
        )

    axes[0].set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm)")
    axes[0].set_xlabel(r"$\theta$")
    axes[1].set_ylabel(r"$\rho_{zz}/\rho_{zz0}$")
    axes[1].set_xlabel(r"$\theta$")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes