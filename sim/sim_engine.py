import csv
import json
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import pandas as pd
import os


class SimulationResult:
    def __init__(self):
        self.results = defaultdict(lambda: {
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "blackjacks": 0,
            "busts": 0,
            "rounds": 0,
            "chip_history": [],
            "starting_chips": 0,
            "bet_history": [],
        })
        self.round_logs = [] 
        self.expert_decisions = []

    def log_round_details(self, player_name, round_num, bet, true_count, chips_before, chips_after, outcome, is_blackjack):
        self.round_logs.append({
            "player": player_name,
            "round": round_num,
            "bet": bet,
            "true_count": true_count,
            "chips_before": chips_before,
            "chips_after": chips_after,
            "delta": chips_after - chips_before,
            "outcome": outcome,
            "blackjack": is_blackjack
        })

    def record_expert_decision(self, player_name, round_num, hand_index, true_count, 
                               hand, dealer_card, action, bet_amount, decks_left):
        same = [d for d in self.expert_decisions
                if d["player"]==player_name
                and d["round"]==round_num
                and d["hand_index"]==hand_index]
        seq = len(same) + 1

        self.expert_decisions.append({
            "player": player_name,
            "round": round_num,
            "hand_index": hand_index,
            "seq": seq,
            "true_count": true_count,
            "player_total": hand.value(),
            "dealer_upcard": dealer_card.value(),
            "is_soft": hand.is_soft(),
            "can_split": hand.can_split(),
            "can_double": hand.can_double(),
            "num_cards": len(hand.cards),
            "decks_left": decks_left,
            "action": action,
            "bet_amount": bet_amount,
            "outcome": None,
            "reward":  None,
        })

    def record_hand(self, player_name, outcome, blackjack=False):
        stats = self.results[player_name]
        if outcome not in ("wins", "losses", "pushes"):
            logging.warning(f"[RECORD_WARNING] Nieznany outcome: {outcome}")
        stats[outcome] += 1
        if blackjack:
            stats["blackjacks"] += 1
        stats["rounds"] += 1

    def record_chips(self, player_name, chips):
        self.results[player_name]["chip_history"].append(chips)

    def record_bet(self, player_name, round_num, bet, chips_before, chips_after, true_count=0.0):
        self.results[player_name]["bet_history"].append({
            "round": round_num,
            "bet": bet,
            "chips_before": chips_before,
            "chips_after": chips_after,
            "true_count": true_count
        })

    def risky_bet_analysis(self, count_threshold=0, bet_multiplier=2):
        data = {}
        for player, stats in self.results.items():
            risky_bets = 0
            total_bets = 0
            avg_bet = pd.DataFrame(stats["bet_history"])["bet"].mean() if stats["bet_history"] else 1

            for bet in stats["bet_history"]:
                if (
                    bet["bet"] >= bet_multiplier * avg_bet and
                    bet["true_count"] <= count_threshold
                ):
                    risky_bets += 1
                total_bets += 1

            data[player] = {
                "risky_bets": risky_bets,
                "total_bets": total_bets,
                "risky_ratio": risky_bets / total_bets if total_bets else 0.0
            }
        return pd.DataFrame.from_dict(data, orient="index")
    
    def update_expert_decision(self, player_name, round_num, hand_index, outcome, reward):
        updated = False
        for entry in reversed(self.expert_decisions):
            if (entry["player"]    == player_name and
                entry["round"]     == round_num   and
                entry["hand_index"]== hand_index  and
                entry["outcome"]   is None):
                entry["outcome"] = outcome
                entry["reward"]  = reward
                updated = True
        if not updated:
            f"[UPDATE_WARNING] Nie znalazłem wpisu do update_expert_decision "
            f"({player_name}, runda={round_num}, ręka={hand_index})"

    def set_starting_chips(self, player_name, amount):
        self.results[player_name]["starting_chips"] = amount

    def to_dict(self):
        return dict(self.results)

    def to_dataframe(self):
        rows = []

        for player, stats in self.results.items():
            chip_history = stats["chip_history"]
            rounds = stats["rounds"] or 1 
            start = stats["starting_chips"]
            end = chip_history[-1] if chip_history else start

            profit = end - start
            ev = profit / rounds
            roi = profit / start if start else 0.0

            # Procenty
            win_pct = stats["wins"] / rounds
            loss_pct = stats["losses"] / rounds
            push_pct = stats["pushes"] / rounds

            # Wariancja z rund (jeśli są dane)
            if chip_history and len(chip_history) > 1:
                changes = pd.Series(chip_history, dtype="float64").diff().dropna()
                variance = changes.var()
            else:
                variance = 0.0

            rows.append({
                "player": player,
                "start_chips": start,
                "end_chips": end,
                "profit": profit,
                "EV_per_round": ev,
                "ROI": roi,
                "win_pct": win_pct,
                "loss_pct": loss_pct,
                "push_pct": push_pct,
                "variance": variance,
                **{k: stats[k] for k in ["wins", "losses", "pushes", "blackjacks", "busts", "rounds"]}
            })

        return pd.DataFrame(rows)


    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

    def save_csv(self, path):
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def plot_summary(self, output_dir="plots"):
        import seaborn as sns
        os.makedirs(output_dir, exist_ok=True)
        df = self.to_dataframe()

        players = df["player"].tolist()
        roi_values = df["ROI"].tolist()
        colors = sns.color_palette("Set2", len(players))
        plt.figure(figsize=(8, 4))
        plt.bar(players, roi_values, color=colors)
        plt.xlabel("Gracz")
        plt.ylabel("ROI")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roi_per_player.png"))
        plt.clf()

        from collections import OrderedDict
        chip_data = OrderedDict()
        for player, stats in self.results.items():
            chip_data[player] = stats.get("chip_history", [])

        max_rounds = max((len(h) for h in chip_data.values()), default=0)
        rounds = list(range(max_rounds))
        for player, hist in list(chip_data.items()):
            if len(hist) < max_rounds:
                last = hist[-1] if hist else 0
                chip_data[player] = hist + [last] * (max_rounds - len(hist))

        plt.figure(figsize=(8, 4))
        palette = sns.color_palette("Set2", len(chip_data))
        for (player, hist), color in zip(chip_data.items(), palette):
            plt.plot(rounds, hist, label=player, color=color)
        plt.xlabel("Runda")
        plt.ylabel("Żetony")
        plt.legend(title="Gracz", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "chip_progression.png"), bbox_inches='tight')
        plt.clf()

        chip_deltas = {
            p: pd.Series(stats["chip_history"]).diff().dropna().reset_index(drop=True)
            for p, stats in self.results.items() if len(stats.get("chip_history", [])) > 1
        }
        if chip_deltas:
            df_deltas = pd.DataFrame(chip_deltas)
            plt.figure(figsize=(8, 4))
            colors = sns.color_palette("Set2", len(df_deltas.columns))

            data = [df_deltas[col].dropna().values for col in df_deltas.columns]

            bp = plt.boxplot(
                data,
                labels=df_deltas.columns,
                patch_artist=True,
                medianprops={'linewidth': 2.5, 'color': 'black'},
                boxprops={'linewidth': 1},
                whiskerprops={'linewidth': 1},
                capprops={'linewidth': 1},
                flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.7}
            )
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            plt.xlabel("")
            plt.ylabel("Zmiana żetonów między rundami")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "chip_variance_boxplot.png"))
            plt.clf()
        
    def save_debug(self, path):
        import csv
        from collections import defaultdict

        max_round = max((dec["round"] for dec in self.expert_decisions), default=0)
        players = list(self.results.keys())

        delta_log = defaultdict(lambda: defaultdict(float))
        bet_log   = defaultdict(lambda: defaultdict(lambda: None))
        tc_log    = defaultdict(lambda: defaultdict(lambda: None))
        for rec in self.round_logs:
            p = rec["player"]
            r = rec["round"]
            delta_log[p][r] += rec["delta"]
            bet_log[p][r]    = rec["bet"]
            tc_log[p][r]     = rec["true_count"]

        cum = defaultdict(int)
        rows = []

        for rnd in range(1, max_round + 1):
            for p in players:
                hist = self.results[p]["chip_history"]
                chips = hist[rnd] if len(hist) > rnd else hist[-1]

                rr = sum((dec["reward"] or 0)
                         for dec in self.expert_decisions
                         if dec["player"] == p and dec["round"] == rnd)
                cum[p] += rr

                dl = delta_log[p].get(rnd, 0)
                bet = bet_log[p].get(rnd)
                tc  = tc_log[p].get(rnd)

                rows.append({
                    "player":      p,
                    "round":       rnd,
                    "bet":         bet,
                    "true_count":  tc,
                    "chips_total": chips,
                    "delta_log":   dl,
                    "round_reward":rr,
                    "cum_reward":  cum[p]
                })

        with open(path, "w", newline="") as f:
            fieldnames = [
              "player","round","bet","true_count",
              "chips_total","delta_log","round_reward","cum_reward"
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def save_decision_dataset(self, path):
        df_dec = pd.DataFrame(self.expert_decisions)
        df_dec = df_dec[df_dec['action'].notnull()].reset_index(drop=True)

        df_logs = pd.DataFrame(self.round_logs)[
            ["player", "round", "bet", "true_count", "chips_before", "chips_after", "delta"]
        ]

        df = df_dec.merge(
            df_logs,
            left_on=["player", "round"],
            right_on=["player", "round"],
            how="left",
            suffixes=("", "_round")
        )

        df.to_csv(path, index=False)


class Simulator:
    def __init__(self, players_config, rounds=1000, rules=None, min_bet=10):
        """
        players_config: lista słowników
            - "name": <string>
            - "chips": <int>
            - albo "strategy_func": funkcja lambda deck -> Strategy(deck)
            - albo "strategy": gotowa instancja strategii (nadpisze .shoe)
        rounds: liczba rund do rozegrania
        rules: obiekt CasinoRules
        min_bet: minimalny zakład
        """
        from collections import defaultdict
        from models.shoe import Shoe
        from game.dealer import Dealer
        from game.player import Player
        from game.table import Table
        from models.chips import Chips
        from game.casino_rules import CasinoRules

        from collections import defaultdict
        from models.shoe import Shoe
        from game.dealer import Dealer
        from game.player import Player
        from game.table import Table
        from models.chips import Chips
        from game.casino_rules import CasinoRules

        self.rounds = rounds
        self.result = SimulationResult()
        self.history = defaultdict(list)

        self.deck = Shoe()
        self.rules = rules or CasinoRules.classic()
        self.dealer = Dealer(self.rules)

        self.table = Table(
            players=[],
            min_bet=min_bet,
            result=self.result,
            strategy_func=None,
            rules=self.rules
        )

        self.players = []
        for conf in players_config:
            name = conf["name"]
            init_chips = conf.get("chips", 100)
            chips_obj = Chips(init_chips)
            strat_func = conf.get("strategy_func")
            strat = conf.get("strategy")

            if strat_func:
                strategy = strat_func(self.deck)
            elif strat:
                strategy = strat
                strategy.shoe = self.deck
            else:
                raise ValueError(f"Player '{name}' must provide a strategy or strategy_func")

            player = Player(name=name, strategy=strategy, chips=chips_obj)
            self.players.append(player)
            self.table.add_player(player)
            self.result.set_starting_chips(name, init_chips)

        self.table.players = self.players

        self.table.expert_logger = next(
            (p.strategy.logger for p in self.players if hasattr(p.strategy, "logger")),
            None
        )
        self.table.deck = self.deck

        self.starting_chips = {
            conf["name"]: conf.get("chips", 100) for conf in players_config
        }

        self.table.expert_logger = next(
            (p.strategy.logger for p in self.players if hasattr(p.strategy, "logger")),
            None
        )
        logging.basicConfig(
            filename="sim_debug.log",
            filemode="w",
            format="%(message)s",
            level=logging.INFO,
        )

    def log_round_start(self, round_num):
        logging.info(f"\n========== Runda {round_num} ==========")
        for p in self.players:
            logging.info(f"{p.name} — żetony: {p.chips.total()}")

    def log_round_end(self):
        for p in self.players:
            logging.info(f"{p.name} — PO rundzie: {p.chips.total()} żetonów")

    def run(self):
        for player in self.players:
            self.result.set_starting_chips(player.name, player.chips.total())
            self.result.record_chips(player.name, player.chips.total())
        for i in range(self.rounds):
            self.log_round_start(i + 1)

            active_players = [
                p for p in self.players
                if not hasattr(p.strategy, "should_play") or p.strategy.should_play()
            ]
            if not active_players:
                logging.info(f"[Runda {i + 1}] Pominięta — brak aktywnych graczy przy odpowiednim True Count")
                continue
            
            chips_before = {p.name: p.chips.total() for p in self.players}
            self.table.play_round()
            
            chips_after = {p.name: p.chips.total() for p in self.players}
            for player in self.players:
                self.result.record_chips(player.name, player.chips.total())

                if hasattr(player, "active_bets") and player.active_bets:
                    bet_amount = player.active_bets[0]
                    tc = self.table.deck.true_count() if hasattr(self.table.deck, "true_count") else 0.0
                    self.result.record_bet(
                        player_name=player.name,
                        round_num=i + 1,
                        bet=bet_amount,
                        chips_before=chips_before[player.name],
                        chips_after=chips_after[player.name],
                        true_count=tc
                    )
                    
                    delta = chips_after[player.name] - chips_before[player.name]
                    if delta > 0:
                        outcome = "win"
                    elif delta < 0:
                        outcome = "loss"
                    else:
                        outcome = "push"

                    is_blackjack = any(h.is_blackjack() for h in player.hands) if hasattr(player, "hands") else False

                    self.result.log_round_details(
                        player_name=player.name,
                        round_num=i + 1,
                        bet=bet_amount,
                        true_count=tc,
                        chips_before=chips_before[player.name],
                        chips_after=chips_after[player.name],
                        outcome=outcome,
                        is_blackjack=is_blackjack
                    )

            
            self.log_round_end()

            for player in self.players:
                delta = chips_after[player.name] - chips_before[player.name]
                logging.info(
                    f"[NET_CHANGE] {player.name}: {chips_before[player.name]} -> {chips_after[player.name]} (delta={delta})"
                )

            if self.table.simulation_ended:
                logging.info("[KONIEC SYMULACJI — brak aktywnych graczy]")
                break


    def report(self):
        df = self.result.to_dataframe()

        for _, row in df.iterrows():
            print(f"\n== {row['player']} ==")
            print(f"Start: {row['start_chips']}, Koniec: {row['end_chips']}, Zysk: {row['profit']}")
            print(f"Rundy: {int(row['rounds'])}")
            print(f"Wygrane: {int(row['wins'])}, Przegrane: {int(row['losses'])}, Remisy: {int(row['pushes'])}")
            print(f"Blackjacki: {int(row['blackjacks'])}, Busty: {int(row['busts'])}")
            print(f"EV/runda: {row['EV_per_round']:.2f}, ROI: {row['ROI']:.2%}")
            print(f"Win%: {row['win_pct']:.1%}, Loss%: {row['loss_pct']:.1%}, Push%: {row['push_pct']:.1%}")
            print(f"Wariancja zysków: {row['variance']:.2f}")
            print(f"Bilans żetonów: {row['end_chips']}")
            print("\n==== ANALIZA RYZYKOWNYCH ZAKŁADÓW ====")
            print("Zakłady >= 2x średni i True Count <= 0:\n")

            risky_df = self.result.risky_bet_analysis()
            print(risky_df.to_string(float_format="%.2f"))

    def plot_results(self, output_dir="plots"):
        self.result.plot_summary(output_dir)

    def save_results(self, json_path="sim_results.json", csv_path="sim_results.csv", 
                     round_log_path="round_log.csv", expert_log_path="expert_decisions.csv"):
        self.result.save_json(json_path)
        self.result.save_csv(csv_path)
        self.result.save_debug("chip_vs_reward.csv")
        risky_df = self.result.risky_bet_analysis()
        risky_df.to_csv("risky_bets.csv")
        pd.DataFrame(self.result.round_logs).to_csv(round_log_path, index=False)
        if self.result.expert_decisions:
            pd.DataFrame(self.result.expert_decisions).to_csv(expert_log_path, index=False)
        self.result.save_decision_dataset("decision_dataset.csv")
