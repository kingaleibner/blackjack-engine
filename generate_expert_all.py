import pandas as pd
import os
from models.shoe import Shoe
from sim.sim_engine import Simulator
from strategies.basic import BasicStrategy
from strategies.hilo import HiLoStrategy
from strategies.adv_count import AdvancedCountStrategy

def run_batch_all(strategies_info, n_runs=100, rounds_per_run=500, initial_chips=1000):
    """
    Uruchamia symulacje z wieloma strategiami jednocześnie.

    strategies_info: lista tupli (name_str, strategy_cls), np. [("basic", BasicStrategy), ...]
    n_runs: ile razy powtarzamy zestaw graczy (dla różnych losowych seedów i reshuffle'ów)
    rounds_per_run: ile rund w jednej symulacji.
    initial_chips: początkowa liczba żetonów dla każdego gracza.
    """
    all_expert = []
    all_logs   = []

    players_config_template = []
    for name_str, strat_cls in strategies_info:
        player_name = name_str.capitalize()
        if name_str.lower() == "ac":
            player_name = "AC"
        elif name_str.lower() == "hilo":
            player_name = "HiLo"
        # inaczej: capitalize
        players_config_template.append({
            "name": player_name,
            "strategy_cls": strat_cls
        })

    for run_id in range(1, n_runs+1):
        print(f"\n--- RUN {run_id}/{n_runs} dla wszystkich strategii ---")

        players_config = []
        for info in players_config_template:
            name = info["name"]
            strat_cls = info["strategy_cls"]
            strategy_func = lambda deck, cls=strat_cls: cls(deck)
            players_config.append({
                "name": name,
                "strategy_func": strategy_func,
                "chips": initial_chips
            })

        sim = Simulator(
            players_config=players_config,
            rounds=rounds_per_run
        )
        sim.run()

        for dec in sim.result.expert_decisions:
            dec["run_id"] = run_id
            player_name = dec.get("player", "")
            strategy_str = player_name.lower()
            dec["strategy"] = strategy_str

        for log in sim.result.round_logs:
            log["run_id"] = run_id
            player_name = log.get("player", "")
            log["strategy"] = player_name.lower()

        all_expert.extend(sim.result.expert_decisions)
        all_logs  .extend(sim.result.round_logs)

    df_expert = pd.DataFrame(all_expert)
    df_logs   = pd.DataFrame(all_logs)

    if 'decks_left' not in df_expert.columns:
        raise RuntimeError("Brakuje kolumny 'decks_left' w expert_decisions")

    df_expert = df_expert[df_expert['action'].notna()].reset_index(drop=True)

    for col in ["player","strategy","run_id","round","bet","true_count","chips_before","chips_after","delta"]:
        if col not in df_logs.columns:
            raise RuntimeError(f"Brakuje kolumny `{col}` w round_logs.")

    df_logs_sel = df_logs[["player","strategy","run_id","round","bet","true_count","chips_before","chips_after","delta"]]

    df_merged = pd.merge(
        df_expert,
        df_logs_sel,
        on=["player","strategy","run_id","round"],
        how="left",
        suffixes=("","_round")
    )

    cols = [
        "player","strategy","run_id","round","hand_index","seq",
        "true_count","decks_left","player_total","dealer_upcard",
        "is_soft","can_split","can_double","num_cards",
        "action","bet_amount","outcome","reward",
        "bet","chips_before","chips_after","delta"
    ]
    cols = [c for c in cols if c in df_merged.columns]
    df_merged = df_merged[cols]

    out_path = "test_dec_all_strategies.csv"
    df_merged.to_csv(out_path, index=False)
    print(f"\nZapisano {len(df_merged)} wierszy do `{out_path}`")
    print(df_merged.head())

    return df_merged

if __name__ == "__main__":
    strategies_info = [
        ("basic", BasicStrategy),
        ("hilo", HiLoStrategy),
        ("ac", AdvancedCountStrategy)
    ]
    n_runs = 500
    rounds_per_run = 500
    initial_chips = 1000

    df_all = run_batch_all(strategies_info, n_runs=n_runs, rounds_per_run=rounds_per_run, initial_chips=initial_chips)
