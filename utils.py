from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


def get_fraction_oracles(
    oracles,
    test_correlation,
    generate_figure=True,
    oracle_label="Oracles",
    test_label="Test correlations",
    fig_name="oracle_fig.png",
):
    """
    Given oracles and test_correlations (both for each neuron), this method
    computes the fraction of oracle performance
    """

    # we will fit a linear function without offset
    def f(x, a):
        return a * x

    if np.any(np.isnan(test_correlation)):
        print(
            "{}% NaNs in test_correlations, NaNs will be set to Zero.".format(
                np.isnan(test_correlation).mean() * 100
            )
        )
    test_correlation[np.isnan(test_correlation)] = 0

    if np.any(np.isinf(test_correlation)):
        print(
            "{}% infinity numbers in test_correlations, infinity will be set to Zero.".format(
                np.isinf(test_correlation).mean() * 100
            )
        )
    test_correlation[np.isinf(test_correlation)] = 0

    if np.any(np.isnan(oracles)):
        print(
            "{}% NaNs in oracles, NaNs will be set to Zero.".format(
                np.isnan(oracles).mean() * 100
            )
        )
    oracles[np.isnan(oracles)] = 0

    if np.any(np.isinf(oracles)):
        print(
            "{}% infinity numbers in oracles, infinity will be set to Zero.".format(
                np.isinf(oracles).mean() * 100
            )
        )
    oracles[np.isinf(oracles)] = 0

    slope, _ = curve_fit(f, oracles, test_correlation)
    format_float = "{:.1f}".format(slope[0] * 100)

    if generate_figure:
        plt.scatter(oracles, test_correlation, s=1, color="orange")
        x = np.linspace(0, 1, 100)
        plt.plot(x, f(x, slope), "r-", label=format_float + "% oracle")
        plt.plot(x, x, "k--", linewidth=1, label="100.0% oracle")
        plt.axvline(x=0, c="gray", linewidth=0.5)
        plt.axhline(y=0, c="gray", linewidth=0.5)
        plt.xlabel(oracle_label)
        plt.ylabel(test_label)
        plt.legend(loc="upper left")
        plt.savefig(fig_name)
        plt.clf()

    return slope
