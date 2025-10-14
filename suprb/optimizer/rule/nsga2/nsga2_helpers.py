import numpy as np
import matplotlib.pyplot as plt
from typing import List

def visualize_pareto_front(self, pareto_front: List, save_path: str = "Paretofront.png") -> None:
    if not pareto_front:
        print("No Pareto front provided for visualization.")
        return

    objs = self._fitness_objs_runtime()
    labels = self._fitness_labels_runtime()

    if len(objs) != 2:
        print(f"Expected exactly 2 objectives, found {len(objs)}. Skipping plot.")
        return

    obj_matrix = np.array([[objs[0](r), objs[1](r)] for r in pareto_front])

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"  # for consistent math font

    plt.figure(figsize=(6, 4))
    plt.scatter(obj_matrix[:, 0], obj_matrix[:, 1], s=40, c="royalblue", edgecolors="black", alpha=0.8)
    plt.xlabel(labels[0], fontsize=11)
    plt.ylabel(labels[1], fontsize=11)
    plt.title("Pareto Front", fontsize=12, pad=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Pareto front visualization saved to '{save_path}'")
