import numpy as np
import torch
from collections import defaultdict
from sim.sim_engine import Simulator
from sim.sim_engine import SimulationResult 


class MultiSimulation:
    def __init__(self, players_config, runs=10, rounds_per_run=1000, rules=None):
        self.players_config = players_config
        self.runs = runs
        self.rounds_per_run = rounds_per_run
        self.rules = rules

        self.final_chips = defaultdict(list)
        self.all_results = []

    def run_all(self):
        for i in range(self.runs):
            print(f"\n===== SYMULACJA {i + 1} / {self.runs} =====")

            for conf in self.players_config:
                strat = conf.get("strategy")
                if hasattr(strat, "set_training_mode"):
                    strat.set_training_mode()

            sim = Simulator(self.players_config, rounds=self.rounds_per_run, rules=self.rules)
            sim.run()
            self.all_results.append(sim.result)

            for conf in self.players_config:
                strat = conf.get("strategy")
                if hasattr(strat, "learn"):
                    strat.learn()

            for player in sim.players:
                self.final_chips[player.name].append(player.chips.total())
                
    def plot_rl_loss(self):
        import matplotlib.pyplot as plt

        for conf in self.players_config:
            strat = conf["strategy"]
            if hasattr(strat, "get_loss_history"):
                losses = strat.get_loss_history()
                if losses:
                    plt.plot(losses, label=conf["name"])

        plt.title("Strata (loss) w czasie treningu")
        plt.xlabel("Epoka / update")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


    def summary(self):
        print("\n===== PODSUMOWANIE WIELU SYMULACJI =====")
        for player_name, chips_list in self.final_chips.items():
            arr = np.array(chips_list)
            avg = arr.mean()
            std = arr.std()
            min_val = arr.min()
            max_val = arr.max()

            print(f"\n== {player_name} ==")
            print(f"Średni bilans: {avg:.2f}")
            print(f"Odchylenie standardowe: {std:.2f}")
            print(f"Min: {min_val}, Max: {max_val}")
            roi = (arr.mean() - 1000) / 1000
            print(f"Średni ROI: {roi:.2%}")

    def generate_combined_plots(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Histogramy
        for player, chips in self.final_chips.items():
            sns.histplot(chips, kde=True, bins=15)
            plt.title(f"Histogram końcowego bilansu: {player}")
            plt.xlabel("Końcowy stan żet.")
            plt.ylabel("Liczba symulacji")
            plt.grid()
            plt.show()

        # Boxplot porównawczy
        df = pd.DataFrame(dict(self.final_chips))
        sns.boxplot(data=df)
        plt.title("Rozrzut końcowego bilansu (źródło: {self.runs} symulacji)")
        plt.ylabel("Końcowy bilans")
        plt.grid()
        plt.show()
