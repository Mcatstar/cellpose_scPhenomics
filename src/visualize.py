"""analyze.py —— 读取 postproc.py 的输出，做分组统计和可视化。

输入（REPORTS_DIR 根目录）:
    global_metrics.csv           每张图的全局汇总
    mito_morphology.csv          每个 mito 实例的形态 + PDM/CM
    ld_morphology.csv            每个 ld 实例的形态 + small/large
    contact_pairs.csv            每对 (mito, ld) 的接触
    contact_width_bins.csv       每张图 × 宽度 bin

输出:
    group_summary.csv            按 C/M/T 分组的均值 ± SEM
    stat_tests.csv               组间检验 + PDM vs CM 检验
    figures/*.png                主要图表

用法:
    python -m src.visualize
    python -m src.visualize --group-col prefix --prefixes C,M,T
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from scipy import stats

from src.config import REPORTS_DIR

app = typer.Typer()

# 统一配色
GROUP_ORDER = ["C", "M", "T"]
GROUP_COLORS = {"C": "#4C72B0", "M": "#DD8452", "T": "#55A868"}


# ==================== 数据加载 ====================

def infer_group(image: str) -> str:
    """从图名推断组别。例如 "C-1-2.5-1" -> "C"。"""
    if not image:
        return "unknown"
    return image.split("-")[0].split("_")[0]


def load_all(report_dir: Path) -> dict[str, pd.DataFrame]:
    """读取所有 CSV，加 group 列。"""
    files = {
        "global": "global_metrics.csv",
        "mito": "mito_morphology.csv",
        "ld": "ld_morphology.csv",
        "pairs": "contact_pairs.csv",
        "bins": "contact_width_bins.csv",
    }
    data: dict[str, pd.DataFrame] = {}
    for key, name in files.items():
        path = report_dir / name
        if not path.is_file():
            logger.warning(f"缺少 {path}")
            data[key] = pd.DataFrame()
            continue
        df = pd.read_csv(path)
        if "image" in df.columns:
            df["group"] = df["image"].map(infer_group)
        data[key] = df
        logger.info(f"{name}: {len(df)} 行")
    return data


# ==================== 分组汇总 ====================

# 每个指标对应的原始列名
GLOBAL_METRICS = [
    "n_mito", "n_ld", "n_ld_small", "n_ld_large",
    "n_pdm", "n_cm", "pdm_ratio",
    "n_mito_contact", "n_ld_contact",
    "n_pattern_single", "n_pattern_multi",
    "total_contact_length_um", "contact_ratio_mito", "contact_ratio_ld",
    "mito_total_area_um2", "mito_mean_area_um2", "mito_total_perimeter_um",
    "mito_mean_circularity", "mito_mean_aspect_ratio", "mito_mean_elongation",
    "ld_total_area_um2", "ld_mean_area_um2", "ld_total_perimeter_um",
]


def group_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """按 group 计算每个指标的 mean/std/sem/median/count。"""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for group, sub in df.groupby("group", sort=False):
            values = sub[metric].dropna()
            if len(values) == 0:
                continue
            rows.append({
                "metric": metric,
                "group": group,
                "n_images": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "sem": float(values.std(ddof=1) / np.sqrt(len(values)))
                       if len(values) > 1 else 0.0,
                "median": float(values.median()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
            })
    return pd.DataFrame(rows)


# ==================== 统计检验 ====================

def test_between_groups(
    df: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    """对所有组两两做 Mann-Whitney U 检验，输出 p 值。"""
    if df.empty:
        return pd.DataFrame()

    rows = []
    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]

    for metric in metrics:
        if metric not in df.columns:
            continue
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                v1 = df.loc[df["group"] == g1, metric].dropna()
                v2 = df.loc[df["group"] == g2, metric].dropna()
                if len(v1) < 2 or len(v2) < 2:
                    continue
                try:
                    stat, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                except ValueError:
                    continue
                # Cohen's d 效应量
                pooled_std = np.sqrt((v1.var(ddof=1) + v2.var(ddof=1)) / 2)
                d = (v1.mean() - v2.mean()) / pooled_std if pooled_std > 0 else 0.0
                rows.append({
                    "metric": metric,
                    "group_a": g1,
                    "group_b": g2,
                    "n_a": len(v1),
                    "n_b": len(v2),
                    "mean_a": float(v1.mean()),
                    "mean_b": float(v2.mean()),
                    "U_statistic": float(stat),
                    "p_value": float(p),
                    "cohens_d": float(d),
                })
    return pd.DataFrame(rows)


def test_pdm_vs_cm(mito_df: pd.DataFrame) -> pd.DataFrame:
    """PDM vs CM 的形态差异（人肝活检方案）。"""
    if mito_df.empty or "class" not in mito_df.columns:
        return pd.DataFrame()

    metrics = ["area_um2", "perimeter_um", "circularity",
               "aspect_ratio", "roundness", "solidity", "elongation"]

    rows = []
    for metric in metrics:
        if metric not in mito_df.columns:
            continue
        pdm = mito_df.loc[mito_df["class"] == "PDM", metric].dropna()
        cm = mito_df.loc[mito_df["class"] == "CM", metric].dropna()
        if len(pdm) < 2 or len(cm) < 2:
            continue
        try:
            stat, p = stats.mannwhitneyu(pdm, cm, alternative="two-sided")
        except ValueError:
            continue
        pooled_std = np.sqrt((pdm.var(ddof=1) + cm.var(ddof=1)) / 2)
        d = (pdm.mean() - cm.mean()) / pooled_std if pooled_std > 0 else 0.0
        rows.append({
            "metric": metric,
            "n_pdm": len(pdm),
            "n_cm": len(cm),
            "mean_pdm": float(pdm.mean()),
            "mean_cm": float(cm.mean()),
            "U_statistic": float(stat),
            "p_value": float(p),
            "cohens_d": float(d),
        })
    return pd.DataFrame(rows)


# ==================== 可视化 ====================

def _setup_plot_style() -> None:
    sns.set_theme(style="ticks", context="talk")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # 中文字体
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def plot_group_boxplots(
    df: pd.DataFrame,
    metrics: list[str],
    titles: list[str],
    out_path: Path,
    ncols: int = 3,
) -> None:
    """每个指标一张箱线图 + 散点，按 group 分组。"""
    if df.empty:
        return
    n = len(metrics)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]

    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in df.columns:
            ax.axis("off")
            continue
        sns.boxplot(
            data=df, x="group", y=metric, order=groups,
            palette=GROUP_COLORS, ax=ax, width=0.6, fliersize=0,
            hue="group", legend=False
        )
        sns.stripplot(
            data=df, x="group", y=metric, order=groups,
            color="black", size=4, alpha=0.6, ax=ax, jitter=True,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # 关掉多余的子图
    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")


def plot_contact_width_spectrum(
    bins_df: pd.DataFrame, out_path: Path
) -> None:
    """接触宽度频谱：0-100 nm 内接触长度分布，按 group 叠加。"""
    if bins_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = [g for g in GROUP_ORDER if g in bins_df["group"].unique()]

    for group in groups:
        sub = bins_df[bins_df["group"] == group]
        if sub.empty:
            continue
        # 按 bin_low_nm 分组，求每个 bin 上所有图的均值
        stats_bin = sub.groupby("bin_low_nm")["contact_length_um"].agg(
            ["mean", "sem", "count"]
        ).reset_index()
        # SEM 标准化到该 bin 的接触长度量级
        stats_bin["sem_filled"] = stats_bin["sem"].fillna(0)

        ax.plot(
            stats_bin["bin_low_nm"], stats_bin["mean"],
            marker="o", label=group, color=GROUP_COLORS[group],
        )
        ax.fill_between(
            stats_bin["bin_low_nm"],
            stats_bin["mean"] - stats_bin["sem_filled"],
            stats_bin["mean"] + stats_bin["sem_filled"],
            color=GROUP_COLORS[group], alpha=0.2,
        )

    ax.set_xlabel("Contact width (nm)")
    ax.set_ylabel("Total contact length (μm)")
    ax.set_title("Mito-LD contact width spectrum")
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")


def plot_pdm_cm_comparison(
    mito_df: pd.DataFrame, out_path: Path
) -> None:
    """PDM vs CM 的形态对比（面积、周长、圆度、长宽比）。"""
    if mito_df.empty or "class" not in mito_df.columns:
        return

    metrics = ["area_um2", "perimeter_um", "circularity",
               "aspect_ratio", "roundness", "solidity"]
    titles = ["Area (μm²)", "Perimeter (μm)", "Circularity",
              "Aspect ratio", "Roundness", "Solidity"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in mito_df.columns:
            ax.axis("off")
            continue
        sns.violinplot(
            data=mito_df, x="class", y=metric, hue="class", legend=False,
            order=["CM", "PDM"], palette={"CM": "#888888", "PDM": "#C44E52"},
            ax=ax, cut=0, inner="box",
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")


def plot_ld_size_distribution(ld_df: pd.DataFrame, out_path: Path) -> None:
    """LD 面积分布（直方图 + KDE），按 group 分组。"""
    if ld_df.empty or "area_um2" not in ld_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    groups = [g for g in GROUP_ORDER if g in ld_df["group"].unique()]

    # 左：直方图
    for group in groups:
        sub = ld_df[ld_df["group"] == group]
        axes[0].hist(
            sub["area_um2"], bins=30, alpha=0.5,
            label=group, color=GROUP_COLORS[group],
        )
    axes[0].set_xlabel("LD area (μm²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("LD area distribution")
    axes[0].legend(title="Group")

    # 右：等效直径箱线
    if "equivalent_diameter_um" in ld_df.columns:
        sns.boxplot(
            data=ld_df, x="group", y="equivalent_diameter_um",
            order=groups, palette=GROUP_COLORS, ax=axes[1],
            fliersize=0, width=0.6,
            hue="group", legend=False
        )
        sns.stripplot(
            data=ld_df, x="group", y="equivalent_diameter_um",
            order=groups, color="black", size=3, alpha=0.5,
            ax=axes[1], jitter=True,
        )
        axes[1].axhline(2.0, color="red", linestyle="--", linewidth=1,
                        label="2 μm 阈值")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Equivalent diameter (μm)")
        axes[1].set_title("LD size by group")
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")


def plot_pdm_ld_correlation(
    global_df: pd.DataFrame, out_path: Path
) -> None:
    """PDM 计数 vs LD 计数（人肝活检方案 r=0.85）。"""
    if global_df.empty:
        return
    if "n_pdm" not in global_df.columns or "n_ld" not in global_df.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    groups = [g for g in GROUP_ORDER if g in global_df["group"].unique()]

    for group in groups:
        sub = global_df[global_df["group"] == group]
        ax.scatter(
            sub["n_ld"], sub["n_pdm"],
            label=group, color=GROUP_COLORS[group],
            s=60, alpha=0.8, edgecolors="black", linewidth=0.5,
        )

    # 拟合线（用所有点）
    if len(global_df) >= 3:
        x = global_df["n_ld"].to_numpy()
        y = global_df["n_pdm"].to_numpy()
        if x.std() > 0:
            r, p = stats.pearsonr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, slope * x_line + intercept,
                    "k--", alpha=0.5, linewidth=1)
            ax.text(
                0.05, 0.95, f"Pearson r={r:.2f}\np={p:.3g}",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
            )

    ax.set_xlabel("LD count")
    ax.set_ylabel("PDM count")
    ax.set_title("PDM vs LD count")
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")


# ==================== 互作接触的绝对长度与相对比例 ====================

def plot_contact_length_per_pair(
    pairs_df: pd.DataFrame, out_path: Path
) -> None:
    """每个接触对的绝对接触长度分布。"""
    if pairs_df.empty or "is_contact" not in pairs_df.columns:
        return
    contact = pairs_df[pairs_df["is_contact"]].copy()
    if contact.empty:
        logger.warning("没有接触对，跳过 contact_length_per_pair")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    groups = [g for g in GROUP_ORDER if g in contact["group"].unique()]

    # 左：箱线 + 散点
    sns.boxplot(
        data=contact, x="group", y="contact_length_um",
        order=groups, palette=GROUP_COLORS, ax=axes[0],
        fliersize=0, width=0.6,
        hue="group", legend=False
    )
    sns.stripplot(
        data=contact, x="group", y="contact_length_um",
        order=groups, color="black", size=3, alpha=0.5,
        ax=axes[0], jitter=True,
    )
    axes[0].set_title(f"Per-pair contact length (n={len(contact)})")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Contact length (μm)")

    # 右：直方图叠加
    for group in groups:
        sub = contact[contact["group"] == group]
        axes[1].hist(
            sub["contact_length_um"], bins=20, alpha=0.5,
            label=f"{group} (n={len(sub)})", color=GROUP_COLORS[group],
        )
    axes[1].set_xlabel("Contact length (μm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of per-pair contact length")
    axes[1].legend(title="Group")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")


def plot_contact_ratio_per_instance(
    mito_df: pd.DataFrame, ld_df: pd.DataFrame, out_path: Path
) -> None:
    """单实例的相对接触比例（只画有接触的实例，contact_ratio > 0）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- 左：Mito contact ratio ----
    if not mito_df.empty and "contact_ratio" in mito_df.columns:
        contact_mito = mito_df[mito_df["contact_ratio"] > 0].copy()
        if not contact_mito.empty:
            groups = [g for g in GROUP_ORDER
                      if g in contact_mito["group"].unique()]
            sns.boxplot(
                data=contact_mito, x="group", y="contact_ratio",
                order=groups, palette=GROUP_COLORS, ax=axes[0],
                fliersize=0, width=0.6,
                hue="group", legend=False
            )
            sns.stripplot(
                data=contact_mito, x="group", y="contact_ratio",
                order=groups, color="black", size=2, alpha=0.4,
                ax=axes[0], jitter=True,
            )
            axes[0].set_title(
                f"Mito contact ratio (n={len(contact_mito)})"
            )
            axes[0].set_xlabel("")
            axes[0].set_ylabel("Contact length / perimeter")
            axes[0].axhline(0, color="gray", linewidth=0.5)
        else:
            axes[0].text(
                0.5, 0.5, "No mito with contact",
                ha="center", va="center", transform=axes[0].transAxes,
            )
            axes[0].axis("off")

    # ---- 右：LD contact ratio ----
    if not ld_df.empty and "contact_ratio" in ld_df.columns:
        contact_ld = ld_df[ld_df["contact_ratio"] > 0].copy()
        if not contact_ld.empty:
            groups = [g for g in GROUP_ORDER
                      if g in contact_ld["group"].unique()]
            sns.boxplot(
                data=contact_ld, x="group", y="contact_ratio",
                order=groups, palette=GROUP_COLORS, ax=axes[1],
                fliersize=0, width=0.6,
                hue="group", legend=False
            )
            sns.stripplot(
                data=contact_ld, x="group", y="contact_ratio",
                order=groups, color="black", size=2, alpha=0.4,
                ax=axes[1], jitter=True,
            )
            axes[1].set_title(
                f"LD contact ratio (n={len(contact_ld)})"
            )
            axes[1].set_xlabel("")
            axes[1].set_ylabel("Contact length / perimeter")
            axes[1].axhline(0, color="gray", linewidth=0.5)
        else:
            axes[1].text(
                0.5, 0.5, "No LD with contact",
                ha="center", va="center", transform=axes[1].transAxes,
            )
            axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")


def plot_pdm_cm_contact_metrics(
    mito_df: pd.DataFrame, out_path: Path
) -> None:
    """PDM vs CM 的接触长度和接触比例对比。"""
    if mito_df.empty or "class" not in mito_df.columns:
        return
    if "contact_length_um" not in mito_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    palette = {"CM": "#888888", "PDM": "#C44E52"}

    # 左：接触长度
    sns.boxplot(
        data=mito_df, x="class", y="contact_length_um",
        order=["CM", "PDM"], palette=palette,
        ax=axes[0], fliersize=0, width=0.6,
        hue="class", legend=False
    )
    sns.stripplot(
        data=mito_df, x="class", y="contact_length_um",
        order=["CM", "PDM"], color="black", size=2, alpha=0.4,
        ax=axes[0], jitter=True,
    )
    axes[0].set_title("Mito contact length by class")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Contact length (μm)")

    # 右：接触比例
    sns.boxplot(
        data=mito_df, x="class", y="contact_ratio",
        order=["CM", "PDM"], palette=palette,
        ax=axes[1], fliersize=0, width=0.6,
        hue="class", legend=False
    )
    sns.stripplot(
        data=mito_df, x="class", y="contact_ratio",
        order=["CM", "PDM"], color="black", size=2, alpha=0.4,
        ax=axes[1], jitter=True,
    )
    axes[1].set_title("Mito contact ratio by class")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Contact length / perimeter")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")


def plot_contact_absolute_vs_relative(
    global_df: pd.DataFrame, out_path: Path
) -> None:
    """每张图：绝对接触长度 vs 相对接触比例。"""
    if global_df.empty:
        return
    needed = ["total_contact_length_um", "contact_ratio_mito"]
    if any(c not in global_df.columns for c in needed):
        return

    fig, ax = plt.subplots(figsize=(6.5, 5))
    groups = [g for g in GROUP_ORDER if g in global_df["group"].unique()]

    for group in groups:
        sub = global_df[global_df["group"] == group]
        ax.scatter(
            sub["total_contact_length_um"],
            sub["contact_ratio_mito"],
            label=group, color=GROUP_COLORS[group],
            s=60, alpha=0.8, edgecolors="black", linewidth=0.5,
        )

    # 拟合线
    if len(global_df) >= 3:
        x = global_df["total_contact_length_um"].to_numpy()
        y = global_df["contact_ratio_mito"].to_numpy()
        if x.std() > 0:
            r, p = stats.pearsonr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, slope * x_line + intercept,
                    "k--", alpha=0.5, linewidth=1)
            ax.text(
                0.05, 0.95, f"Pearson r={r:.2f}\np={p:.3g}",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
            )

    ax.set_xlabel("Total contact length (μm)  [absolute]")
    ax.set_ylabel("Contact length / mito perimeter  [relative]")
    ax.set_title("Absolute vs relative contact by image")
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"saved {out_path.name}")

    
# ==================== 主流程 ====================
@app.command()
def main(
    report_dir: Path = REPORTS_DIR,
    figure_dir: Path = REPORTS_DIR / "figures",
) -> None:
    """读取 postproc 输出，做分组统计和可视化。"""
    _setup_plot_style()
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(report_dir)
    global_df = data["global"]
    mito_df = data["mito"]
    ld_df = data["ld"]
    pairs_df = data["pairs"]
    bins_df = data["bins"]

    # ---- 分组汇总 ----
    summary = group_summary(global_df, GLOBAL_METRICS)
    summary.to_csv(report_dir / "group_summary.csv", index=False)
    logger.info(f"group_summary.csv: {len(summary)} 行")

    # ---- 组间检验 ----
    between = test_between_groups(global_df, GLOBAL_METRICS)
    between.to_csv(report_dir / "stat_tests_between_groups.csv", index=False)
    logger.info(f"stat_tests_between_groups.csv: {len(between)} 行")

    # ---- PDM vs CM ----
    pdm_cm = test_pdm_vs_cm(mito_df)
    pdm_cm.to_csv(report_dir / "stat_tests_pdm_vs_cm.csv", index=False)
    logger.info(f"stat_tests_pdm_vs_cm.csv: {len(pdm_cm)} 行")

    # ---- 可视化：原有 ----
    plot_group_boxplots(
        global_df,
        metrics=[
            "n_mito", "n_ld", "n_pdm", "pdm_ratio",
            "total_contact_length_um", "contact_ratio_mito",
        ],
        titles=[
            "Mito count", "LD count", "PDM count", "PDM ratio",
            "Total contact length (μm)", "Contact ratio (mito)",
        ],
        out_path=figure_dir / "global_metrics_boxplots.png",
        ncols=3,
    )

    plot_group_boxplots(
        global_df,
        metrics=[
            "mito_mean_area_um2", "mito_total_perimeter_um",
            "mito_mean_circularity", "mito_mean_aspect_ratio",
            "ld_mean_area_um2", "ld_total_perimeter_um",
        ],
        titles=[
            "Mean mito area (μm²)", "Mito total perimeter (μm)",
            "Mito circularity", "Mito aspect ratio",
            "Mean LD area (μm²)", "LD total perimeter (μm)",
        ],
        out_path=figure_dir / "morphology_boxplots.png",
        ncols=3,
    )

    plot_contact_width_spectrum(bins_df, figure_dir / "contact_width_spectrum.png")
    plot_pdm_cm_comparison(mito_df, figure_dir / "pdm_vs_cm.png")
    plot_ld_size_distribution(ld_df, figure_dir / "ld_size_distribution.png")
    plot_pdm_ld_correlation(global_df, figure_dir / "pdm_vs_ld_count.png")

    # ---- 可视化：新增（绝对长度 + 相对比例）----
    plot_contact_length_per_pair(
        pairs_df, figure_dir / "contact_length_per_pair.png"
    )
    plot_contact_ratio_per_instance(
        mito_df, ld_df, figure_dir / "contact_ratio_per_instance.png"
    )
    plot_pdm_cm_contact_metrics(
        mito_df, figure_dir / "pdm_cm_contact_metrics.png"
    )
    plot_contact_absolute_vs_relative(
        global_df, figure_dir / "contact_absolute_vs_relative.png"
    )

    logger.success(f"分析完成，结果输出到 {report_dir}")


if __name__ == "__main__":
    app()