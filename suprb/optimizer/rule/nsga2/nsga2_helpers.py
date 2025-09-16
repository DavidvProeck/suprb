from typing import List

import numpy as np
from matplotlib import pyplot as plt

#deprecated
#TODO: Rework to be sklearn conform. Extra objectives are currently added during runtime to be conform with sklearn standards. Need to fix this function.

def visualize_pareto_front(
        self,
        pareto_front,
):
    x = [self.fitness_objs[0](r) for r in pareto_front]
    y = [self.fitness_objs[1](r) for r in pareto_front]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=30, edgecolors='k')
    plt.xlabel(self.fitness_objs_labels[0])
    plt.ylabel(self.fitness_objs_labels[1])
    plt.title('Pareto Optimal Set (Front 0)')
    plt.tight_layout()
    plt.savefig("Paretofront.png", dpi=150)
    plt.close()
