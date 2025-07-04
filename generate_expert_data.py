import pandas as pd
import os
from models.shoe import Shoe
from sim.sim_engine import Simulator
from strategies.basic import BasicStrategy
from strategies.hilo import HiLoStrategy
from strategies.adv_count import AdvancedCountStrategy
from strategies.random import RandomProfileStrategy
from game.profiles import PlayerProfile

def run_batch(strategy_cls, name: str, profile_style: str, n_runs=100, rounds_per_run=500):
    """
    Uruchamia n_runs symulacji po rounds_per_run rund każda, dla jednej strategii.
    strategy_cls: klasa strategii oczekująca __init__(shoe, profile, ...)
    name: nazwa strategii (używana w outputach i kolumnie 'strategy')
    profile_style: styl profilu, np. 'cautious', 'aggressive' lub 'random'
    """
    all_expert = []
    all_logs   = []

    profile = PlayerProfile(name=name, style=profile_style)

    for i in range(n_runs):
        run_id = i + 1
        print(f"\n--- {name.upper()} — Symulacja {run_id}/{n_runs} ---")
        sim = Simulator(
            players_config=[{
                "name": name.capitalize(),
                "strategy_func": lambda deck, cls=strategy_cls, profile=profile: cls(deck, profile),
                "chips": 1000
            }],
            rounds=rounds_per_run
        )
        sim.run()

        for dec in sim.result.expert_decisions:
            dec["run_id"] = run_id
        for log in sim.result.round_logs:
            log["run_id"] = run_id

        all_expert.extend(sim.result.expert_decisions)
        all_logs  .extend(sim.result.round_logs)

    df_dec = pd.DataFrame(all_expert)
    if 'decks_left' not in df_dec.columns:
        raise RuntimeError("Brakuje kolumny 'decks_left'")

    df_dec = df_dec[df_dec['action'].notnull()].reset_index(drop=True)

    df_logs = pd.DataFrame(all_logs)[
        ["player", "round", "run_id", "bet", "true_count", "chips_before", "chips_after", "delta"]
    ]

    df = df_dec.merge(
        df_logs,
        on=["player", "round", "run_id"],
        how="left",
        suffixes=("", "_round")
    )
    df["strategy"] = name

    cols = [
        "player", "strategy", "run_id", "round", "hand_index", "seq",
        "true_count", "decks_left", "player_total", "dealer_upcard",
        "is_soft", "can_split", "can_double", "num_cards",
        "action", "bet_amount", "outcome", "reward",
        "bet", "chips_before", "chips_after", "delta"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    out_path = f"test_dec_{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[{name.upper()}] Zapisano {len(df)} wierszy ➝ `{out_path}`")
    print(df.head())

if __name__ == "__main__":
    run_batch(RandomProfileStrategy, name="rd", profile_style="cautious", n_runs=500, rounds_per_run=500)
