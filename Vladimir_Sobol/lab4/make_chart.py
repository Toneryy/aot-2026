from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "metrics.csv"
OUT = ROOT / "results" / "chart.png"

SYSTEMS = ["version_1", "version_2 без повтора", "итоговая система"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DPI = 200
BAR_W = 0.24
BAR_GAP = 0.03
RADIUS_PT = 3

SHARE_ROWS = [
    ("Валидный итоговый ответ", "Валидный\nитоговый ответ"),
    ("Precision micro", "Precision\nmicro"),
    ("Recall micro", "Recall\nmicro"),
    ("F1 micro", "F1\nmicro"),
]


def comma(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


def rounded_bar(ax, x, width, height, color):
    if height <= 0:
        return
    origin = ax.transData.transform((0, 0))
    unit = ax.transData.transform((1, 1)) - origin
    r_px = RADIUS_PT * DPI / 72
    rx = min(r_px / unit[0], width / 2)
    ry = min(r_px / unit[1], height)
    k = 0.5523
    x0, x1 = x - width / 2, x + width / 2
    verts = [
        (x0, 0), (x0, height - ry),
        (x0, height - ry + k * ry), (x0 + rx - k * rx, height), (x0 + rx, height),
        (x1 - rx, height),
        (x1 - rx + k * rx, height), (x1, height - ry + k * ry), (x1, height - ry),
        (x1, 0), (x0, 0),
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def style_axis(ax, labels, ylim, ticks, fmt):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", length=0, colors=INK_2, labelsize=9.5)
    ax.tick_params(axis="x", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_zorder(4)


def draw_groups(ax, values, label_fmt):
    step = BAR_W + BAR_GAP
    for g, row in enumerate(values):
        for s, value in enumerate(row):
            x = g + (s - 1) * step
            rounded_bar(ax, x, BAR_W, value, COLORS[s])
            ax.annotate(label_fmt(value), (x, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=INK, zorder=5)


def main():
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    df = pd.read_csv(CSV, index_col=0)[SYSTEMS]

    totals = df.loc["Извлечений всего"].astype(int).tolist()
    count_rows = [
        ("Evidence нет в тексте дословно", "Evidence нет\nв тексте дословно"),
        ("Извлечений без опоры в evidence",
         "Извлечений без опоры\nв evidence\n(из " + " / ".join(map(str, totals)) + " извлечений)"),
    ]

    fig = plt.figure(figsize=(11, 5.2), dpi=DPI, facecolor=SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.3], left=0.06, right=0.985,
                            top=0.78, bottom=0.2, wspace=0.18)
    ax_share = fig.add_subplot(grid[0])
    ax_count = fig.add_subplot(grid[1])

    share = df.loc[[key for key, _ in SHARE_ROWS]].values
    style_axis(ax_share, [label for _, label in SHARE_ROWS], (0, 1.08),
               [0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda v, _: "0" if v == 0 else comma(v, 1))
    draw_groups(ax_share, share, lambda v: comma(v, 2))

    counts = df.loc[[key for key, _ in count_rows]].values
    top = counts.max() * 1.15
    style_axis(ax_count, [label for _, label in count_rows], (0, top),
               list(range(0, int(top) + 1, 50)), lambda v, _: f"{int(v)}")
    draw_groups(ax_count, counts, lambda v: f"{int(round(v))}")

    ax_share.set_title("Доли, от 0 до 1", loc="left", fontsize=10.5, color=INK_2, pad=10)
    ax_count.set_title("Количество, шт.", loc="left", fontsize=10.5, color=INK_2, pad=10)

    fig.suptitle("Сравнение version_1 и итоговой системы на 50 отзывах", x=0.06, y=0.965,
                 ha="left", fontsize=14, fontweight="bold", color=INK)
    handles = [Patch(facecolor=c, edgecolor="none", label=n) for c, n in zip(COLORS, SYSTEMS)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.915), ncol=3,
               frameon=False, fontsize=10, labelcolor=INK, handlelength=1.1, handleheight=1.1,
               columnspacing=1.8, handletextpad=0.5)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
