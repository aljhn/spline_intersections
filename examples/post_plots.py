import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from spline_intersections.splines import get_spline, spline_eval


def setup_style():
    sns.set_theme(
        style="darkgrid", 
        context="notebook",
        font_scale=1.2)
    sns.set_palette("deep")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text"],
        "mathtext.fontset": "stix",

        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,

        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,

        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def setup_figure():
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    return fig, ax


def plot1_1():
    # f = lambda t: -0.05 * t * t + 0.5 * t - 5.0 * np.exp(-0.5 * t) + 5.0
    # df = lambda t: -0.1 * t + 0.5 + 2.5 * np.exp(-0.5 * t)
    # t = np.array([0.0, 2.0, 4.5, 10.0, 14.5])

    f = lambda t: -0.05 * t * t + t - 0.5 * t * np.exp(0.5 * (t - 15.0))
    df = lambda t: -0.1 * t + 1.0 - (0.5 + 0.25 * t) * np.exp(0.5 * (t - 15.0))
    t = np.array([0.0, 4.5, 6.5, 10.0, 12.5])

    p = f(t)
    v = df(t)

    dt = 1.0

    t_dense = np.linspace(0.0, 14.0, 100)
    p_dense = f(t_dense)

    fig, ax = setup_figure()

    ax.plot(
        t_dense,
        p_dense,
        color="C0",
        linewidth=2,
        zorder=1
    )
    ax.scatter(
        t,
        p,
        s=150,
        color="C0",
        edgecolor="white",
        linewidth=1.5,
        zorder=3
    )
    ax.quiver(
        t,
        p,
        dt * np.ones_like(t),
        v * dt,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
        headwidth=4.0,
        headlength=4.0,
        headaxislength=4.0,
        color="black",
        zorder=2
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$f(t)$")
    ax.margins(x=0.05, y=0.05)
    sns.despine(ax=ax)

    fig.savefig(
        "plots/plot_1_1.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("Finished: plots/plot_1_1.png")


def plot1_2():
    mult = 10.0

    t0 = 0.0
    p0 = np.array([0.0, 0.0])
    v0 = np.array([-1.0, 1.0]) * mult

    t1 = 1.0
    p1 = np.array([4.0, 4.0])
    v1 = np.array([-1.0, -1.0]) * mult

    t2 = 2.0
    p2 = np.array([6.0, -2.0])
    v2 = np.array([0.0, 1.0]) * mult

    t3 = 3.0
    p3 = np.array([10.0, 2.0])
    v3 = np.array([-1.0, 1.0]) * mult

    spline0 = get_spline(t0, p0, v0, t1, p1, v1)
    tspan0 = np.linspace(t0, t1, 100)
    x0 = spline_eval(spline0, tspan0)

    spline1 = get_spline(t1, p1, v1, t2, p2, v2)
    tspan1 = np.linspace(t1, t2, 100)
    x1 = spline_eval(spline1, tspan1)

    spline2 = get_spline(t2, p2, v2, t3, p3, v3)
    tspan2 = np.linspace(t2, t3, 100)
    x2 = spline_eval(spline2, tspan2)

    fig, ax = setup_figure()

    ax.scatter(
        [p0[0], p1[0], p2[0], p3[0]],
        [p0[1], p1[1], p2[1], p3[1]],
        s=150,
        color="black",
        edgecolor="white",
        linewidth=1.5,
        zorder=2
    )
    ax.plot(x0[0, :], x0[1, :], linewidth=3)
    ax.plot(x1[0, :], x1[1, :], linewidth=3)
    ax.plot(x2[0, :], x2[1, :], linewidth=3)

    for t, p in zip([t0, t1, t2, t3], [p0, p1, p2, p3]):
        ax.annotate(
            f"$t={t}$",
            xy=p,
            xytext=(15, -5),
            textcoords="offset points",
        )

    arrow_scale = 0.15
    for p, v in [(p0, v0), (p1, v1), (p2, v2), (p3, v3)]:
        ax.quiver(
            p[0],
            p[1],
            v[0] * arrow_scale,
            v[1] * arrow_scale,
            angles="xy",
            scale_units="xy",
            color="black",
            scale=1,
            width=0.005,
            headwidth=4.0,
            headlength=4.0,
            headaxislength=4.0,
            zorder=3
        )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim([-2, 12])
    ax.set_ylim([-4, 6])
    ax.margins(x=0.05, y=0.05)
    ax.set_aspect("equal")
    sns.despine(ax=ax)

    fig.savefig(
        "plots/plot_1_2.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("Finished: plots/plot_1_2.png")

def plot2_1():
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, subplot_kw={"projection": "3d"})

    t = np.linspace(0.0, 5.0, 100)
    ax.plot(t - t * t + 10.0, t - t * t / 10.0, 1.0 + t / 20.0, color="C0", linewidth=3)

    t = np.linspace(-5.0, 5.0, 100)
    ax.plot(2.0 * t, np.exp(t / 2) - 1.0, t, color="C1", linewidth=3)

    t = np.linspace(-5.0, 0.0, 100)
    ax.plot(t - t * t + 10.0, t - t * t / 10.0, 1.0 + t / 20.0, color="C0", linewidth=3)

    ax.view_init(elev=30, azim=-60)

    ax.margins(0.05)
    sns.despine(ax=ax)
    fig.savefig(
        "plots/plot_2_1.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("Finished: plots/plot_2_1.png")

def plot2_2():
    fig, ax = setup_figure()

    t = np.linspace(-4.0, 5.0, 200)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    f = 0.1 * t5 - 0.3 * t4 - 1.1 * t3 + 2.7 * t2 + t - 2.4
    # f = 0.1 * (t + 1.0) * (t - 1.0) * (t - 2.0) * (t + 3.0) * (t - 4.0)

    ax.plot(t, f, color="C0", linewidth=3)
    ax.plot((-10.0, 10.0), (0.0, 0.0), color="black", linewidth=1)
    ax.plot((0.0, 0.0), (-10.0, 10.0), color="black", linewidth=1)

    ax.scatter(
        (-1, 1, 2, -3, 4),
        (0, 0, 0, 0, 0),
        s=100,
        color="C0",
        edgecolor="white",
        linewidth=1.5,
        zorder=3
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$f(t)$")
    ax.set_xlim([-4.5, 5.5])
    ax.set_ylim([-6, 9])
    ax.margins(x=0.05, y=0.05)
    sns.despine(ax=ax)

    fig.savefig(
        "plots/plot_2_2.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("Finished: plots/plot_2_2.png")

def plot2_3():
    fig, ax = setup_figure()

    t0 = np.array([1.0, 2.5, 3.0, 5.0])
    t1 = np.array([1.5, 2.75, 3.5])

    h0 = 0.5
    h1 = -0.5

    def draw_spline(p0, v0, p1, v1, color):
        spline = get_spline(0.0, p0, v0, 1.0, p1, v1)
        t = np.linspace(0.0, 1.0, 100)
        f = spline_eval(spline, t)
        ax.plot(f[0, :], f[1, :], color=color, linewidth=3)

    draw_spline(np.array([t0[0], h0]), np.array([5.0, 2.0]), np.array([t0[1], h0]), np.array([1.0, 1.0]), "C0")
    draw_spline(np.array([t0[1], h0]), np.array([1.0, 1.0]), np.array([t0[2], h0]), np.array([0.0, -2.0]), "C0")
    draw_spline(np.array([t0[2], h0]), np.array([0.0, -2.0]), np.array([t0[3], h0]), np.array([0.0, -5.0]), "C0")

    draw_spline(np.array([t1[0], h1]), np.array([2.0, 4.0]), np.array([t1[1], h1]), np.array([1.0, 2.0]), "C1")
    draw_spline(np.array([t1[1], h1]), np.array([1.0, 2.0]), np.array([t1[2], h1]), np.array([0.5, -1.0]), "C1")

    ax.scatter(
        t0,
        np.ones_like(t0) * h0,
        s=150,
        color="C0",
        edgecolor="white",
        linewidth=1.5,
        zorder=3
    )

    ax.scatter(
        t1,
        np.ones_like(t1) * h1,
        s=150,
        color="C1",
        edgecolor="white",
        linewidth=1.5,
        zorder=3
    )

    ax.plot((t0[0], t0[0]), (h0 + 0.5, h0 - 0.5), color="black", linewidth=2)
    ax.plot((t0[1], t0[1]), (h0 + 0.5, h0 - 0.5), color="black", linewidth=2)
    ax.plot((t0[2], t0[2]), (h0 + 0.5, h0 - 0.5), color="black", linewidth=2)
    ax.plot((t0[3], t0[3]), (h0 + 0.5, h0 - 0.5), color="black", linewidth=2)

    ax.plot((t1[0], t1[0]), (h1 + 0.5, h1 - 0.5), color="black", linewidth=2)
    ax.plot((t1[1], t1[1]), (h1 + 0.5, h1 - 0.5), color="black", linewidth=2)
    ax.plot((t1[2], t1[2]), (h1 + 0.5, h1 - 0.5), color="black", linewidth=2)

    ax.plot((t0[0], t0[0]), (h1 + 0.5, h1 - 0.5), color="black", linewidth=2, linestyle="dashed")
    ax.plot((t0[1], t0[1]), (h1 + 0.5, h1 - 0.5), color="black", linewidth=2, linestyle="dashed")
    ax.plot((t0[2], t0[2]), (h1 + 0.5, h1 - 0.5), color="black", linewidth=2, linestyle="dashed")
    ax.plot((t0[3], t0[3]), (h1 + 0.5, h1 - 0.5), color="black", linewidth=2, linestyle="dashed")

    ax.plot((t1[0], t1[0]), (h0 + 0.5, h0 - 0.5), color="black", linewidth=2, linestyle="dashed")
    ax.plot((t1[1], t1[1]), (h0 + 0.5, h0 - 0.5), color="black", linewidth=2, linestyle="dashed")
    ax.plot((t1[2], t1[2]), (h0 + 0.5, h0 - 0.5), color="black", linewidth=2, linestyle="dashed")

    ts = (t0[0], t1[0], t0[1], t1[1], t0[2], t1[2], t0[3])
    for i in range(len(ts) - 1):
        u0 = ts[i]
        u1 = ts[i + 1]

        x = [u0, u0, u1, u1]
        y = [h0 + 0.5, h1 - 0.5, h1 - 0.5, h0 + 0.5]

        if i % 2 == 0:
            c = "C0"
        else:
            c = "C1"
        ax.fill(x, y, color=c, alpha=0.2)

    ax.plot((-100, 100), (0, 0), color="black", linewidth=1)

    ax.set_xlim([0, 6])
    ax.set_ylim([-1.5, 1.5])
    ax.margins(x=0.05, y=0.05)
    sns.despine(ax=ax)

    fig.savefig(
        "plots/plot_2_3.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("Finished: plots/plot_2_3.png")


def main():
    if not os.path.exists("plots"):
        os.mkdir("plots")

    plot1_1()
    plot1_2()

    plot2_1()
    plot2_2()
    plot2_3()


if __name__ == "__main__":
    main()
