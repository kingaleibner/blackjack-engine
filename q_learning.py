import os
import pickle
import random
import math
from collections import defaultdict
from itertools import groupby

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sim.sim_engine import Simulator
from strategies.strategy_base import StrategyBase
from matplotlib.patches import Patch

PRETRAIN_EPISODES = 20000    # pełne użycie logów pre‑training
MIXING_EPISODES  = 200000     # czas mieszania z malejącą wagą

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.05,
                 epsilon_decay=None, total_episodes=1_000_000,
                 batch_size=1000):
        self.alpha_start   = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        if epsilon_decay is None:
            steps = max(1, total_episodes // batch_size)
            self.epsilon_decay = (epsilon_min/epsilon_start)**(1/steps)
        else:
            self.epsilon_decay = epsilon_decay

        self.Q       = defaultdict(lambda: defaultdict(float))
        self.actions = ["hit","stand","double","split","surrender"]
        self.state_counts = defaultdict(int)

    def discretize(self, tc, total, dealer_up, is_soft, num_cards, can_split):
        """
        Dyskretyzacja stanu:
         - tc: true_count (float) -> zaokrąglony i obcięty do [-10,10]
         - total: suma gracza (int) -> bucket [4..21]
         - dealer_up: wartość karty krupiera (2..11)
         - is_soft: bool
         - num_cards: liczba kart w ręce (int) -> 0 jeśli 2 karty, 1 jeśli >2
         - can_split: bool -> 1 jeśli możliwy split, 0 w przeciwnym wypadku
        Zwraca tuple stanu: (tc_b, total_b, dealer_b, soft_b, nc_b, split_b)
        """
        try:
            total_i = int(total)
        except:
            total_i = 4
        total_b  = min(max(total_i, 4), 21)
        try:
            dealer_i = int(dealer_up)
        except:
            dealer_i = 2
        dealer_b = min(max(dealer_i, 2), 11)
        try:
            tc_rounded = int(round(float(tc)))
        except:
            tc_rounded = 0
        tc_b = min(max(tc_rounded, -10), 10)
        soft_b   = 1 if is_soft else 0
        try:
            nc = int(num_cards)
        except:
            nc = 2
        nc_b     = 0 if nc == 2 else 1
        split_b  = 1 if can_split else 0
        return (tc_b, total_b, dealer_b, soft_b, nc_b, split_b)

    def select_action(self, state, valid_actions=None):
        pool = valid_actions or self.actions
        if random.random() < self.epsilon or not any(a in self.Q[state] for a in pool):
            return random.choice(pool)
        qs = self.Q[state]
        valid_with_values = [a for a in pool if a in qs]
        if not valid_with_values:
            return random.choice(pool)
        best_q = max(qs[a] for a in valid_with_values)
        best_actions = [a for a in valid_with_values if qs[a] == best_q]
        return random.choice(best_actions)

    def update_q(self, s, a, r, s_next, valid_next, ep):
        """
        Standardowy update Q-learning:
         Q(s,a) <- Q(s,a) + alpha_t * [r + gamma * max_a' Q(s_next, a') - Q(s,a)]
        alpha_t dynamicznie maleje: alpha_start / sqrt(ep)
        """
        alpha_t = self.alpha_start / math.sqrt(ep) if ep > 0 else self.alpha_start
        q_sa    = self.Q[s][a]
        if valid_next:
            max_next = max(self.Q[s_next].get(a2, 0.0) for a2 in valid_next)
        else:
            max_next = 0.0
        new_q = q_sa + alpha_t * (r + self.gamma * max_next - q_sa)
        self.Q[s][a] = new_q
        self.state_counts[s] += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class RLStrategy(StrategyBase):
    def __init__(self, shoe, agent, min_bet: float = 10.0):
        super().__init__()
        self.shoe    = shoe
        self.agent   = agent
        self.min_bet = min_bet

    def bet_size(self, min_bet: float = None) -> float:
        mb = min_bet if min_bet is not None else self.min_bet
        tc = float(self.shoe.true_count())
        if tc >= 3:
            return mb * 3
        elif tc >= 2:
            return mb * 2
        elif tc >= 2:
            return mb * 1.5
        else:
            return mb

    def decide(self, hand, dealer_card):
        tc = self.shoe.true_count()
        pt = hand.value()
        du = dealer_card.value()
        is_soft = hand.is_soft()
        num_cards = len(hand.cards)
        can_split = hand.can_split() and self.shoe.rules.allow_split
        state = self.agent.discretize(tc, pt, du, is_soft, num_cards, can_split)
        valid = ['hit','stand']
        if hand.can_double() and self.shoe.rules.allow_double:
            valid.append('double')
        if can_split:
            valid.append('split')
        if hand.can_surrender() and self.shoe.rules.allow_surrender:
            valid.append('surrender')
        return self.agent.select_action(state, valid)


def evaluate_agent(agent, rounds=1000, trials=5):
    evs = []
    for _ in range(trials):
        sim = Simulator(
            players_config=[{
                "name": "RL",
                "strategy_func": lambda d, ag=agent: RLStrategy(d, ag),
                "chips": 1000
            }],
            rounds=rounds
        )
        sim.run()
        df = sim.result.to_dataframe()
        evs.append(df.loc[df['player'] == "RL", "EV_per_round"].iloc[0])
    return sum(evs)/len(evs)


def plot_learning_curves(dfm, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    palette = sns.color_palette('Set2', 3)

    # EV na rundę
    plt.figure(figsize=(6,4))
    plt.plot(dfm['episode'], dfm['EV'], color=palette[0])
    plt.xlabel('Epizody')
    plt.ylabel('EV na rundę')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'learning_curve_EV.png'))
    plt.clf()

    # epsilon
    plt.figure(figsize=(6,4))
    plt.plot(dfm['episode'], dfm['epsilon'], color=palette[1])
    plt.xlabel('Epizody')
    plt.ylabel('ε')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'learning_curve_epsilon.png'))
    plt.clf()

    # liczba stanów Q
    plt.figure(figsize=(6,4))
    plt.plot(dfm['episode'], dfm['states'], color=palette[2])
    plt.xlabel('Epizody')
    plt.ylabel('Liczba stanów Q')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'learning_curve_states.png'))
    plt.clf()


def plot_state_visit_histogram(agent, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    counts = list(agent.state_counts.values())
    if not counts:
        return
    plt.figure(figsize=(6,4))
    plt.hist(counts, bins=50, color=sns.color_palette('Blues')[5])
    plt.xlabel('Liczba aktualizacji Q dla stanu')
    plt.ylabel('Liczba stanów')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'histogram_stanow.png'))
    plt.clf()


def plot_policy_heatmap(agent, tc_value, is_soft, can_split_flag, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    actions = ["hit","stand","double","split","surrender"]
    action_to_int = {a: i for i, a in enumerate(actions)}
    action_to_letter = {'hit':'H','stand':'S','double':'D','split':'P','surrender':'U'}
    palette = sns.color_palette('Set2', len(actions))

    if is_soft:
        totals = list(range(13,22))
    else:
        totals = list(range(4,22))

    dealers = list(range(2,12))

    grid = np.full((len(totals), len(dealers)), np.nan)
    for i, tot in enumerate(totals):
        for j, du in enumerate(dealers):
            state = agent.discretize(tc_value, tot, du, bool(is_soft), 2, can_split_flag)
            qs = agent.Q.get(state, {})
            if qs:
                best = max(qs, key=lambda a: qs[a])
                grid[i,j] = action_to_int.get(best, np.nan)

    import matplotlib.colors as mcolors
    cmap_obj = mcolors.ListedColormap(palette)
    masked_grid = np.ma.masked_invalid(grid)

    plt.figure(figsize=(6,5))
    plt.imshow(masked_grid, origin='lower', cmap=cmap_obj, vmin=0, vmax=len(actions)-1)
    plt.xticks(ticks=np.arange(len(dealers)), labels=dealers)
    plt.yticks(ticks=np.arange(len(totals)), labels=totals)
    plt.xlabel('Karta krupiera')
    plt.ylabel('Suma gracza')
    title_flag = "soft" if is_soft else "hard"
    split_flag_str = "split_ok" if can_split_flag else "no_split"

    for i in range(len(totals)):
        for j in range(len(dealers)):
            val = grid[i,j]
            if np.isnan(val):
                continue
            idx = int(val)
            letter = action_to_letter[actions[idx]]
            rgb = palette[idx]
            lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            text_color = 'white' if lum < 0.5 else 'black'
            plt.text(j, i, letter, ha='center', va='center', color=text_color, fontsize=6)

    patches = []
    for a, idx in action_to_int.items():
        color = palette[idx]
        letter = action_to_letter[a]
        patches.append(Patch(facecolor=color, label=f"{letter}: {a}"))
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    fname = f'policy_tc{tc_value}_{title_flag}_{split_flag_str}.png'
    plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight')
    plt.clf()


def plot_Q_progression(agent_history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_actions = len(next(iter(agent_history.values())).columns) - 1
    palette = sns.color_palette('Set2', n_actions)
    for state, dfst in agent_history.items():
        state_str = "_".join(str(x) for x in state)
        plt.figure(figsize=(6,4))
        for idx, action in enumerate([c for c in dfst.columns if c!='episode']):
            plt.plot(dfst['episode'], dfst[action], label=action, color=palette[idx])
        plt.xlabel('Epizody')
        plt.ylabel('Q(s,a)')
        plt.legend(fontsize=6)
        plt.tight_layout()
        fname = f'Q_progression_{state_str}.png'
        plt.savefig(os.path.join(out_dir, fname))
        plt.clf()


def train_q_learning(n_episodes=1_000_000,
                     batch_size=1000,
                     save_path="q_table.pkl",
                     expert_log_paths=None,
                     plots_dir="plots_q_learning",
                     track_states=None):
    random.seed(42)
    agent = QLearningAgent(total_episodes=n_episodes,
                           batch_size=batch_size)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            qdict = pickle.load(f)
        agent.Q = defaultdict(lambda: defaultdict(float), qdict)
        # print(f"Wczytano Q‑tabelę ({len(agent.Q)} stanów).")

    expert_sequences = []
    if expert_log_paths:
        df_logs = pd.concat((pd.read_csv(p) for p in expert_log_paths), ignore_index=True)
        df_logs.sort_values(['run_id','round','seq'], inplace=True)
        for _, group in df_logs.groupby(['run_id','round']):
            expert_sequences.append(group.to_dict('records'))
        # print(f"Załadowano {len(expert_sequences)} sekwencji z logów ekspertów.")

    metrics = []
    agent_history = {}
    if track_states:
        for s in track_states:
            agent_history[s] = []

    for start in range(1, n_episodes+1, batch_size):
        runs = min(batch_size, n_episodes - (start-1))
        ep = start
        sim = Simulator(
            players_config=[{
                'name': 'QAgent',
                "strategy_func": lambda deck, ag=agent: RLStrategy(deck, ag, min_bet=10),
                'chips': 1000
            }],
            rounds=runs
        )
        sim.run()

        reward_map = {
            (dec['player'], dec['round'], dec['hand_index']): dec['reward']
            for dec in sim.result.expert_decisions
            if dec['player']=="QAgent" and dec['reward'] is not None
        }

        seqs = [d for d in sim.result.expert_decisions if d['player'] == 'QAgent']
        seqs.sort(key=lambda d: (d['round'], d['hand_index'], d['seq']))
        for (rnd, hand_idx), grp in groupby(seqs, key=lambda d: (d['round'], d['hand_index'])):
            seq = list(grp)
            for i, dec in enumerate(seq):
                # a) stan
                s = agent.discretize(
                    dec['true_count'],
                    dec['player_total'],
                    dec['dealer_upcard'],
                    dec['is_soft'],
                    dec['num_cards'],
                    dec.get('can_split', False)
                )
                r = reward_map.get((dec['player'], dec['round'], dec['hand_index']), 0.0) \
                    if i == len(seq) - 1 else 0.0
                if i + 1 < len(seq):
                    nxt = seq[i + 1]
                    s_next = agent.discretize(
                        nxt['true_count'],
                        nxt['player_total'],
                        nxt['dealer_upcard'],
                        nxt['is_soft'],
                        nxt['num_cards'],
                        nxt.get('can_split', False)
                    )
                    valid_next = ['hit', 'stand']
                    if nxt.get('can_double', False):    valid_next.append('double')
                    if nxt.get('can_split', False):     valid_next.append('split')
                    if nxt.get('can_surrender', False): valid_next.append('surrender')
                else:
                    s_next, valid_next = s, []
                agent.update_q(s, dec['action'], r, s_next, valid_next, ep)

        if expert_sequences:
            if ep <= PRETRAIN_EPISODES:
                seqs_to_use = expert_sequences
            elif ep <= PRETRAIN_EPISODES + MIXING_EPISODES:
                frac = 1 - (ep - PRETRAIN_EPISODES) / MIXING_EPISODES
                count = max(1, int(frac * len(expert_sequences)))
                seqs_to_use = random.sample(expert_sequences, count)
            else:
                seqs_to_use = []
            for seq in seqs_to_use:
                for i, rec in enumerate(seq):
                    s = agent.discretize(rec['true_count'], rec['player_total'], rec['dealer_upcard'],
                                         rec['is_soft'], rec['num_cards'], rec.get('can_split', False))
                    a = rec['action']
                    r = seq[-1]['delta'] if i == len(seq)-1 else 0.0
                    if i+1 < len(seq):
                        nxt = seq[i+1]
                        s_next = agent.discretize(nxt['true_count'], nxt['player_total'], nxt['dealer_upcard'],
                                                  nxt['is_soft'], nxt['num_cards'], nxt.get('can_split', False))
                        valid_next = ['hit','stand']
                        if nxt.get('can_double', False):    valid_next.append('double')
                        if nxt.get('can_split', False):     valid_next.append('split')
                        if nxt.get('can_surrender', False): valid_next.append('surrender')
                    else:
                        s_next, valid_next = s, []
                    agent.update_q(s, a, r, s_next, valid_next, ep)

        agent.decay_epsilon()
        ev = evaluate_agent(agent)
        metrics.append({'episode': ep, 'EV': ev, 'states': len(agent.Q), 'epsilon': agent.epsilon})
        print(f"[Ep{ep}] ε={agent.epsilon:.4f}, stany={len(agent.Q)}, EV={ev:.3f}")

        if track_states:
            for s in track_states:
                row = {'episode': ep}
                for a in agent.actions:
                    row[a] = agent.Q.get(s, {}).get(a, 0.0)
                agent_history[s].append(row)

    with open(save_path, 'wb') as f:
        pickle.dump(dict(agent.Q), f)
    dfm = pd.DataFrame(metrics)
    os.makedirs(plots_dir, exist_ok=True)
    dfm.to_csv(os.path.join(plots_dir, 'debug_metrics.csv'), index=False)

    plot_learning_curves(dfm, plots_dir)
    plot_state_visit_histogram(agent, plots_dir)
    for tc in [-2, 0, 2, 5]:
        for soft in [0,1]:
            plot_policy_heatmap(agent, tc_value=tc, is_soft=soft, can_split_flag=False, out_dir=plots_dir)
            plot_policy_heatmap(agent, tc_value=tc, is_soft=soft, can_split_flag=True, out_dir=plots_dir)

    if track_states:
        hist_dfs = {}
        for s, rows in agent_history.items():
            dfst = pd.DataFrame(rows)
            hist_dfs[s] = dfst
        plot_Q_progression(hist_dfs, plots_dir)

    print(f"Zapisano Q‑tabelę i wykresy w folderze: {plots_dir}")
    return agent


if __name__ == "__main__":
    track = [
        (0, 13, 6, 0, 0, 0),
        (2, 16, 10, 0, 0, 0),
        (5, 12, 4, 0, 0, 0),
    ]
    agent = train_q_learning(
        n_episodes=1_000_000,
        batch_size=1000,
        save_path="q_table.pkl",
        expert_log_paths=[
            "test_dec_ac.csv",
            "test_dec_all_strategies.csv",
            "test_dec_basic.csv",
            "test_dec_hilo.csv"
        ],
        plots_dir="plots_q_learning",
        track_states=track
    )
    from strategies.basic import BasicStrategy
    from strategies.hilo import HiLoStrategy
    from strategies.adv_count import AdvancedCountStrategy
    sim = Simulator(
        players_config=[
            {"name":"BS","strategy_func": lambda d: BasicStrategy(d),"chips":1000},
            {"name":"HiLo","strategy_func": lambda d: HiLoStrategy(d),"chips":1000},
            {"name":"AC","strategy_func": lambda d: AdvancedCountStrategy(d),"chips":1000},
            {"name":"RL","strategy_func": lambda d,ag=agent: RLStrategy(d,ag),"chips":1000},
        ],
        rounds=1000
    )
    sim.run()
    sim.report()
    sim.plot_results("plots_rl_vs_others")
