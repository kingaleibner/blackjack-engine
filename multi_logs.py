import os
import pandas as pd

from sim.sim_engine import Simulator
from strategies.basic import BasicStrategy
from strategies.hilo import HiLoStrategy
from strategies.adv_count import AdvancedCountStrategy
from strategies.q_learning import QLearningStrategy
from strategies.random import RandomProfileStrategy
from game.profiles import PlayerProfile

def run_simulation(
    scenario_name,
    player_strategies,
    n_runs=200,
    rounds_per_run=500,
    initial_chips=1000,
    out_dir="logs"
):
    """
    player_strategies: lista dictów:
      - name: etykieta gracza (np. "BS", "QL")
      - strategy_func: funkcja deck -> instancja strategii
    """
    os.makedirs(out_dir, exist_ok=True)
    all_expert = []
    all_logs   = []

    for run_id in range(1, n_runs+1):
        players_config = [
            {
                "name": p["name"],
                "strategy_func": p["strategy_func"],
                "chips": initial_chips
            }
            for p in player_strategies
        ]
        sim = Simulator(players_config=players_config, rounds=rounds_per_run)
        sim.run()

        for dec in sim.result.expert_decisions:
            dec["run_id"]   = run_id
            dec["strategy"] = dec["player"].lower()
        for log in sim.result.round_logs:
            log["run_id"]   = run_id
            log["strategy"] = log["player"].lower()

        all_expert.extend(sim.result.expert_decisions)
        all_logs.extend(sim.result.round_logs)

    df_expert = pd.DataFrame(all_expert)
    df_logs   = pd.DataFrame(all_logs)

    df_expert = df_expert[df_expert["action"].notna()].reset_index(drop=True)

    df_last = (
        df_expert
        .sort_values("seq")
        .groupby(["player","strategy","run_id","round"], as_index=False)
        .tail(1)
    )

    log_cols = [
        "player","strategy","run_id","round",
        "bet","true_count","chips_before","chips_after","delta","outcome"
    ]
    df_logs_sel = df_logs[log_cols]

    df_merged = pd.merge(
        df_last,
        df_logs_sel,
        on=["player","strategy","run_id","round"],
        how="left"
    )

    out_path = os.path.join(out_dir, f"{scenario_name}.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"Zapisano {len(df_merged)} wierszy do {out_path}")


if __name__ == "__main__":
    profile_cautious   = PlayerProfile(style="cautious")
    profile_aggressive = PlayerProfile(style="aggressive")
    profile_random     = PlayerProfile(style="random")
    profile_hilo       = PlayerProfile(style="hilo")

    scenarios = [
        ("sim1_BS_HiLo_AC", [
            {"name":"BS",   "strategy_func": lambda deck: BasicStrategy(deck)},
            {"name":"HiLo", "strategy_func": lambda deck: HiLoStrategy(deck)},
            {"name":"AC",   "strategy_func": lambda deck: AdvancedCountStrategy(deck)},
        ]),
        ("sim2_BS_HiLo_QL_cautious", [
            {"name":"BS",   "strategy_func": lambda deck: BasicStrategy(deck)},
            {"name":"HiLo", "strategy_func": lambda deck: HiLoStrategy(deck)},
            {"name":"QL",   "strategy_func": lambda deck: QLearningStrategy(deck, profile_cautious, q_table_path="q_table.pkl")},
        ]),
        ("sim3_BS_HiLo_RD_cautious", [
            {"name":"BS",   "strategy_func": lambda deck: BasicStrategy(deck)},
            {"name":"HiLo", "strategy_func": lambda deck: HiLoStrategy(deck)},
            {"name":"RD",   "strategy_func": lambda deck: RandomProfileStrategy(deck, profile_cautious)},
        ]),
        ("sim4_BS_QL_aggressive_RD_aggressive", [
            {"name":"BS", "strategy_func": lambda deck: BasicStrategy(deck)},
            {"name":"QL", "strategy_func": lambda deck: QLearningStrategy(deck, profile_aggressive, q_table_path="q_table.pkl")},
            {"name":"RD", "strategy_func": lambda deck: RandomProfileStrategy(deck, profile_aggressive)},
        ]),
        ("sim5_BS_QL_cautious_RD_cautious", [
            {"name":"BS", "strategy_func": lambda deck: BasicStrategy(deck)},
            {"name":"QL", "strategy_func": lambda deck: QLearningStrategy(deck, profile_cautious, q_table_path="q_table.pkl")},
            {"name":"RD", "strategy_func": lambda deck: RandomProfileStrategy(deck, profile_cautious)},
        ]),
        ("sim6_BS_QL_random_RD_random", [
            {"name":"BS", "strategy_func": lambda deck: BasicStrategy(deck)},
            {"name":"QL", "strategy_func": lambda deck: QLearningStrategy(deck, profile_random, q_table_path="q_table.pkl")},
            {"name":"RD", "strategy_func": lambda deck: RandomProfileStrategy(deck, profile_random)},
        ]),
        ("sim7_HiLo_QL_hilo_AC", [
            {"name":"HiLo","strategy_func": lambda deck: HiLoStrategy(deck)},
            {"name":"QL",  "strategy_func": lambda deck: QLearningStrategy(deck, profile_hilo, q_table_path="q_table.pkl")},
            {"name":"AC",  "strategy_func": lambda deck: AdvancedCountStrategy(deck)},
        ]),
        ("sim8_BS_HiLo_AC_QL_cautious_RD_cautious", [
            {"name":"BS",   "strategy_func": lambda deck: BasicStrategy(deck)},
            {"name":"HiLo", "strategy_func": lambda deck: HiLoStrategy(deck)},
            {"name":"AC",   "strategy_func": lambda deck: AdvancedCountStrategy(deck)},
            {"name":"QL",   "strategy_func": lambda deck: QLearningStrategy(deck, profile_cautious, q_table_path="q_table.pkl")},
            {"name":"RD",   "strategy_func": lambda deck: RandomProfileStrategy(deck, profile_cautious)},
        ]),
        ("sim9_QL_cautious_aggressive_random", [
            {"name":"QL_c", "strategy_func": lambda deck: QLearningStrategy(deck, profile_cautious, q_table_path="q_table.pkl")},
            {"name":"QL_a", "strategy_func": lambda deck: QLearningStrategy(deck, profile_aggressive, q_table_path="q_table.pkl")},
            {"name":"QL_r", "strategy_func": lambda deck: QLearningStrategy(deck, profile_random, q_table_path="q_table.pkl")},
        ]),
    ]

    n_runs = 200
    rounds_per_run = 500
    initial_chips = 1000

    for scenario_name, players in scenarios:
        print(f"Rozpoczynam {scenario_name}")
        run_simulation(
            scenario_name,
            players,
            n_runs=n_runs,
            rounds_per_run=rounds_per_run,
            initial_chips=initial_chips,
            out_dir="logs"
        )
