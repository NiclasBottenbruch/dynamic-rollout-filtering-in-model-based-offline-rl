import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr


def plot_correlation_bars(doc:dict, uncertainty_measures:list, error_key:str='model_error_l2', title:str="", print_corr_values:bool=True, fig_size:tuple=(10, 6)):
    """
    Plots a bar chart comparing the Pearson and Spearman correlation coefficients of different uncertainty measures with the model error.
    Bars are ordered by (pearson + spearman) in decreasing order.
    """
    pearson_corrs = []
    spearman_corrs = []

    for u in uncertainty_measures:
        pearson_r = np.corrcoef(doc[u], doc[error_key])[0, 1]
        rank_r, _ = spearmanr(doc[u], doc[error_key])
        pearson_corrs.append(pearson_r)
        spearman_corrs.append(rank_r)
        if print_corr_values:
            measure_appendix = " "*(30-len(u))
            print(f"Uncertainty Measure: {u}{measure_appendix} pearson corr: {pearson_r:3f} spearman rank corr: {rank_r:3f}")

    # Order by sum of pearson + spearman, decreasing
    order = np.argsort(-(np.array(pearson_corrs) + np.array(spearman_corrs)))
    uncertainty_measures_sorted = [uncertainty_measures[i] for i in order]
    pearson_corrs_sorted = [pearson_corrs[i] for i in order]
    spearman_corrs_sorted = [spearman_corrs[i] for i in order]

    x = np.arange(len(uncertainty_measures_sorted))
    width = 0.35

    plt.figure(figsize=fig_size)
    bars1 = plt.bar(x - width/2, pearson_corrs_sorted, width, label='Pearson r', color='dodgerblue')
    bars2 = plt.bar(x + width/2, spearman_corrs_sorted, width, label='Spearman rank r', color='orange')
    plt.xticks(x, uncertainty_measures_sorted, rotation=30, ha='right')
    plt.ylabel('Correlation r')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Annotate bars with values
    for bar in bars1:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}", 
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
    for bar in bars2:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}", 
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')

    plt.show()







def plot_scatter_correlation(x, y, xlabel='X-axis', ylabel='Y-axis', title='Scatter Correlation Plot', mark_percentile:int=70, bins=1000, fig_size=(8, 6)):
    """
    Plots a scatter plot of x vs y and displays the Spearman correlation coefficient.

    Args:
        x (array-like): The x values.
        y (array-like): The y values.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the plot.
        fig_size (tuple): The figure size.
    """

    pairwise_median = np.median(x)
    pairwise_percentile = np.percentile(x, mark_percentile)
    model_error_median = np.median(y)
    model_error_70th = np.percentile(y, mark_percentile)

    plt.figure(figsize=fig_size)
    # Jointplot with marginal histograms
    g = sns.jointplot(
        x=x,
        y=y,
        kind="scatter",
        marginal_kws=dict(bins=bins, fill=True),
        s=2,
        alpha=0.1
    )
    g.fig.set_size_inches(*fig_size)
    g.ax_joint.set_xscale("log")
    g.ax_joint.set_yscale("log")
    g.ax_joint.set_xlabel(f"{xlabel} (log scale)")
    g.ax_joint.set_ylabel(f"{ylabel} (log scale)")
    g.fig.suptitle(title, y=1.03)

    # Mark median and mark_percentile percentile on marginal histograms
    for ax, median, perc, axis in [
        (g.ax_marg_x, pairwise_median, pairwise_percentile, 'x'),
        (g.ax_marg_y, model_error_median, model_error_70th, 'y')
    ]:
        if axis == 'x':
            ax.axvline(median, color='r', linestyle='--', label='Median')
            ax.axvline(perc, color='g', linestyle='--', label=f'{mark_percentile}th percentile')
        else:
            ax.axhline(median, color='r', linestyle='--', label='Median')
            ax.axhline(perc, color='g', linestyle='--', label=f'{mark_percentile}th percentile')
        ax.legend(loc="upper right")

    # Draw thin lines across the scatter plot for median and mark_percentile percentile
    for value, color, label in [
        (pairwise_median, 'r', f'{pairwise_median:.2e}'),
        (pairwise_percentile, 'g', f'{pairwise_percentile:.2e}')
    ]:
        g.ax_joint.axvline(value, color=color, linestyle='dotted', linewidth=0.5)
        g.ax_joint.text(value, g.ax_joint.get_ylim()[1], label, color=color, va='top', ha='right', fontsize=8, rotation=90)

    for value, color, label in [
        (model_error_median, 'r', f'{model_error_median:.2e}'),
        (model_error_70th, 'g', f'{model_error_70th:.2e}')
    ]:
        g.ax_joint.axhline(value, color=color, linestyle='dotted', linewidth=0.5)
        g.ax_joint.text(g.ax_joint.get_xlim()[1], value, label, color=color, va='bottom', ha='right', fontsize=8)

    # Calculate correlations
    correlation = np.corrcoef(x, y)[0, 1]
    spearman_corr, _ = spearmanr(x, y)

    # Print values on plot
    textstr = f"Pearson r: {correlation:.3f}\nSpearman rank r: {spearman_corr:.3f}"
    g.ax_joint.text(
        0.98, 0.02, textstr,
        transform=g.ax_joint.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    plt.show()



def plot_filtering_analysis(doc, filter_indicator, bins=5000, fig_size=(22, 9), filter_criterion:str=None):
    all_data = [doc["model_error_l2"], doc["model_error_l2"][filter_indicator == 0], doc["model_error_l2"][filter_indicator == 1]]
    xmin = min([np.min(d[d > 0]) for d in all_data])  # avoid log(0)
    xmax = max([np.max(d) for d in all_data])

    data_all = doc["model_error_l2"]
    data_0 = doc["model_error_l2"][filter_indicator == 0]
    data_1 = doc["model_error_l2"][filter_indicator == 1]
    portion_0 = len(data_0) / len(doc["model_error_l2"])
    portion_1 = len(data_1) / len(doc["model_error_l2"])

    model_error = doc["model_error_l2"]
    filtered = (filter_indicator == 1)

    thresholds = np.logspace(-5, np.log10(np.max(model_error)), num=200, base=10)
    portions = []
    for t in thresholds:
        mask = model_error > t
        if np.sum(mask) == 0:
            portions.append(0)
        else:
            portions.append(np.sum(filtered & mask) / np.sum(mask))

    bar_thresholds = [1e-3, 1e-2, 1e-1, 1]
    bar_portions = []
    for t in bar_thresholds:
        mask = model_error > t
        if np.sum(mask) == 0:
            bar_portions.append(0)
        else:
            bar_portions.append(np.sum(filtered & mask) / np.sum(mask))

    # --- Plot subplots ---
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    # Set overall title including filter criterion if provided
    if filter_criterion is not None:
        fig.suptitle(f"Filtering Analysis - Filter Criterion: {filter_criterion}", fontsize=16, y=1.05)
    else:
        fig.suptitle("Filtering Analysis", fontsize=16, y=1.05)

    # First plot: histograms
    axs[0].hist(data_all[data_all > 0], bins=bins, alpha=1, label="All data", color="darkgray", density=True)
    axs[0].hist(data_0[data_0 > 0], bins=bins, alpha=0.5, label=f"Accepted, {portion_0:.2%}", color="green", density=True)
    axs[0].hist(data_1[data_1 > 0], bins=bins, alpha=0.5, label=f"Filtered, {portion_1:.2%}", color="red", density=True)
    for data, color, label in [
        (data_all, "gray", "All"),
        (data_0, "darkgreen", "Accepted"),
        (data_1, "darkred", "Filtered")
    ]:
        mean = np.mean(data)
        axs[0].axvline(mean, color=color, linestyle='--', linewidth=1, label=f"{label} Mean: {mean:.3f}")
    axs[0].set_xscale("log")
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_xlabel("Model Error L2 (log scale)")
    axs[0].set_ylabel("Percentage (normalized density)")
    axs[0].set_title("Comparison of Model Error L2 distributions (normalized)")
    axs[0].legend()
    axs[0].grid(True, which="both", linestyle=":", alpha=0.3)

    # Second plot: portion filtered curve and bars
    axs[1].plot(thresholds, portions, color="orange")
    axs[1].fill_between(thresholds, portions, color="lightblue", alpha=0.5)
    axs[1].bar(bar_thresholds, bar_portions, width=np.array(bar_thresholds)*0.15+1e-4, color="dodgerblue", alpha=0.8, zorder=3)
    for x, y in zip(bar_thresholds, bar_portions):
        axs[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color="black")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Model Error L2 (log scale)")
    axs[1].set_ylabel("Portion filtered")
    axs[1].set_title("Portion of filtered data above Model Error L2 Values")
    axs[1].set_ylim(0, 1)
    axs[1].grid(True, which="both", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.show()