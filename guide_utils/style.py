"""One look for every figure. Call style.use() at the top of a notebook."""
import matplotlib.pyplot as plt

INK, TEXT, MUTED, LINE, SOFT = "#171c20", "#3b454c", "#69747c", "#d5dbdf", "#f3f6f7"
TEAL, ORANGE, BLUE, PURPLE, RED, GREEN, GREY = "#0E8A7B", "#C2611A", "#2E6FAD", "#7E3FBE", "#B3382E", "#1F7A45", "#9AA3AB"
PALETTE = [TEAL, ORANGE, BLUE, PURPLE, RED, GREEN, GREY]


def use():
    plt.rcParams.update({
        "figure.figsize": (7.6, 3.4), "figure.dpi": 110, "savefig.facecolor": "white", "font.size": 11, "text.color": INK,
        "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=PALETTE),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "lines.linewidth": 2.0, "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })


def bare(ax):
    """Remove ticks and frame: for images."""
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
