"""论文图只消费已经核验的比较数据。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.reporting import sanitize_task_id


def _label(task: str) -> str:
    name = task.split("/", 1)[-1]
    return {"enthalpy_of_vaporization_or_sublimation": "Enthalpy of vaporization",
            "pec50": "pEC50", "transfer_organic": "Organic transfer"}.get(name, name.replace("_", " ").capitalize())


def _save(fig: Any, destination: Path) -> None:
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def write_paper_figures(payload: dict[str, Any], scatters: dict[Any, Any], output: Path) -> None:
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "ilume-paper-v1",
                         "pdf.fonttype": 42}):
        for split, section in payload["sections"].items():
            destination = output / split
            overall = section["overall"]
            colors = ["#003f88" if r["model_id"] == "ilume" else "#aab5c0" for r in overall]
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.barh([r["model"] for r in overall], [r["macro_normalized_mae"] for r in overall], color=colors)
            ax.invert_yaxis()
            ax.set_xlabel("Task-equal MAE / common training-set standard deviation (lower is better)")
            ax.set_title("Five-fold validation" if split == "valid" else "Five-model test ensemble")
            for index, row in enumerate(overall):
                ax.annotate(f"{row['macro_normalized_mae']:.4f}",
                            (row["macro_normalized_mae"], index), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=9)
            ax.set_xlim(right=max(r["macro_normalized_mae"] for r in overall) * 1.18 or 1.0)
            _save(fig, destination / "overall")

            gains = section["gains"]
            fig, ax = plt.subplots(figsize=(10, max(4, len(gains) * 0.3)))
            for index, row in enumerate(gains):
                value = row["mae_reduction_percent"]
                if value is None:
                    ax.text(0, index, "Undefined: baseline MAE = 0", va="center", fontsize=8)
                    continue
                color = "#003f88" if value >= 0 else "#c26a33"
                ax.plot([0, value], [index, index], color=color, linewidth=2)
                ax.scatter([value], [index], color=color, s=25)
                ax.annotate(f"{value:+.1f}%", (value, index), xytext=(5 if value >= 0 else -5, 0),
                            textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=8)
            ax.axvline(0, color="#666666", linewidth=0.8)
            ax.set_yticks(range(len(gains)), [_label(r["task"]) for r in gains])
            ax.invert_yaxis()
            ax.margins(x=0.23)
            ax.set_xlabel("ILUME MAE reduction vs. the best of ten baselines for each task (%)")
            ax.set_title("All tasks in registry order; negative values retained")
            _save(fig, destination / "task_gains")

            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = plt.get_cmap("tab20")
            for index, model in enumerate(r["model_id"] for r in overall):
                rows = [r for r in section["performance_profile"] if r["model_id"] == model]
                ax.step([r["relative_mae_threshold"] for r in rows], [r["task_fraction"] for r in rows],
                        where="post", label=rows[0]["model"], color="#003f88" if model == "ilume" else cmap(index),
                        linewidth=2.8 if model == "ilume" else 1.1)
            ax.set_xscale("log")
            ax.set_xlim(left=1)
            ax.set_ylim(0, 1.03)
            ax.set_xlabel("MAE / best MAE among all eleven models (log scale)")
            ax.set_ylabel("Fraction of tasks")
            ax.set_title("Performance profile over the complete task set")
            ax.legend(loc="lower right", fontsize=8, ncol=2)
            _save(fig, destination / "performance_profile")

            scatter_root = destination / "ilume_scatter"
            scatter_root.mkdir()
            for task in (r["task"] for r in gains):
                points = np.asarray(scatters[(split, task)])
                low, high = float(points.min()), float(points.max())
                margin = (high - low) * 0.04 or max(1.0, abs(low) * 0.04)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(points[:, 0], points[:, 1], s=max(2, min(16, 80 / len(points) ** 0.3)),
                           c="#003f88", alpha=0.5, edgecolors="none", rasterized=True)
                ax.plot([low - margin, high + margin], [low - margin, high + margin], color="#c26a33", linewidth=1)
                ax.set(xlim=(low - margin, high + margin), ylim=(low - margin, high + margin),
                       xlabel="Observed (original units)", ylabel="Predicted (original units)", title=_label(task))
                ax.set_aspect("equal", adjustable="box")
                ax.text(0.03, 0.97, f"ILUME | {'pooled validation rows' if split == 'valid' else 'test ensemble'}\nn = {len(points)}",
                        transform=ax.transAxes, va="top", fontsize=9)
                _save(fig, scatter_root / sanitize_task_id(task))


__all__ = ["write_paper_figures"]
