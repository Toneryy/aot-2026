import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.dossier import RESULTS

COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
NAMES = {"A": "A: полный контекст", "B": "B: BM25 top-15", "C": "C: p-adic контекст"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def panel(ax, groups, data, title):
    modes = list(COLORS)
    width = 0.26
    for i, m in enumerate(modes):
        xs = [g + (i - 1) * (width + 0.02) for g in range(len(groups))]
        vals = [data[m][k] for k in groups]
        bars = ax.bar(xs, vals, width=width, color=COLORS[m], label=NAMES[m], zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.02, f"{v + 1e-9:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups.values()), fontsize=9, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0,25", "0,5", "0,75", "1"], fontsize=8.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_facecolor(SURFACE)


def main():
    mm = pd.read_csv(RESULTS / "metrics.csv").set_index("mode")
    stab = pd.read_csv(RESULTS / "stability.csv").set_index("mode")
    base = {m: {"accuracy": mm.loc[m, "accuracy"], "ev_f1": mm.loc[m, "ev_f1"],
                "cev_recall": mm.loc[m, "cev_recall"]} for m in COLORS}
    stress = {m: {"perm": 1 - stab.loc[m, "perm_changed_share"], "remove": stab.loc[m, "remove_expected_share"],
                  "contra": stab.loc[m, "contra_both_share"]} for m in COLORS}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor=SURFACE)
    panel(axes[0], {"accuracy": "Точность\nстатуса", "ev_f1": "Evidence F1", "cev_recall": "Полнота\ncounterevidence"},
          base, "Основной прогон, 12 вопросов")
    panel(axes[1], {"perm": "Статус не изменился\nпосле перестановки", "remove": "insufficient\nпосле удаления",
                    "contra": "both после\nпротиворечия"}, stress, "Стресс-тесты")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS / "chart.png", dpi=200, facecolor=SURFACE)
    print("results/chart.png")


if __name__ == "__main__":
    main()
