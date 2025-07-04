import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

action_colors = {
    'H': '#f28e8e',  # Hit
    'S': '#8ef2a0',  # Stand
    'D': '#87cdee',  # Double
    'R': '#a28ef2',  # Surrender
    'P': '#f2b78e',  # Split
}

dealer_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']

hard_hands = {
    8:  ['H'] * 10,
    9:  ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
    10: ['D'] * 8 + ['H', 'H'],
    11: ['D'] * 9 + ['H'],
    12: ['H', 'H', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
    13: ['S'] * 5 + ['H'] * 5,
    14: ['S'] * 5 + ['H'] * 5,
    15: ['S'] * 5 + ['H', 'H', 'R', 'R', 'R'],
    16: ['S'] * 5 + ['H', 'H', 'R', 'R', 'R'],
    17: ['S'] * 10,
    18: ['S'] * 10,
    19: ['S'] * 10,
    20: ['S'] * 10,
}

soft_hands = {
    'A2': ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
    'A3': ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
    'A4': ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
    'A5': ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
    'A6': ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
    'A7': ['S', 'D', 'D', 'D', 'D', 'S', 'S', 'H', 'H', 'H'],
    'A8': ['S'] * 10,
    'A9': ['S'] * 10,
}

split_hands = {
    'A,A': ['P'] * 10,
    '10,10': ['S'] * 10,
    '9,9': ['P', 'P', 'P', 'P', 'P', 'S', 'P', 'P', 'S', 'S'],
    '8,8': ['P'] * 10,
    '7,7': ['P'] * 6 + ['H'] * 4,
    '6,6': ['P'] * 5 + ['H'] * 5,
    '5,5': ['D'] * 8 + ['H', 'H'],
    '4,4': ['H', 'H', 'H', 'P', 'P', 'H', 'H', 'H', 'H', 'H'],
    '3,3': ['P'] * 5 + ['H'] * 5,
    '2,2': ['P'] * 5 + ['H'] * 5,
}

def get_rgb_color_grid(df, color_map):
    color_grid = np.empty(df.shape + (3,), dtype=float)
    for i, row in enumerate(df.values):
        for j, val in enumerate(row):
            hex_color = color_map.get(val, "#ffffff")
            rgb = mcolors.to_rgb(hex_color)
            color_grid[i, j] = rgb
    return color_grid

def draw_colored_strategy(df, title, filename):
    rgb_grid = get_rgb_color_grid(df, action_colors)
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(df) + 2))
    # ax.set_title(title, fontsize=16, pad=20)

    for y, row in enumerate(df.index):
        for x, col in enumerate(df.columns):
            action = df.loc[row, col]
            color = rgb_grid[y, x]
            rect = plt.Rectangle([x, y], 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.5, action, ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax.set_yticks(np.arange(len(df.index)) + 0.5)
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.set_xlabel("Karta krupiera", fontsize=15)
    ax.set_ylabel("Ręka gracza", fontsize=15)
    ax.set_xlim(0, len(df.columns))
    ax.set_ylim(0, len(df.index))

    legend_elements = [
        Patch(facecolor=action_colors['H'], label='Hit'),
        Patch(facecolor=action_colors['S'], label='Stand'),
        Patch(facecolor=action_colors['D'], label='Double Down'),
        Patch(facecolor=action_colors['R'], label='Surrender'),
        Patch(facecolor=action_colors['P'], label='Split'),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

hard_df = pd.DataFrame.from_dict(hard_hands, orient='index', columns=dealer_cards).reindex(sorted(hard_hands.keys()))
soft_df = pd.DataFrame.from_dict(soft_hands, orient='index', columns=dealer_cards).reindex(sorted(soft_hands.keys()))
split_df = pd.DataFrame.from_dict(split_hands, orient='index', columns=dealer_cards).reindex(sorted(split_hands.keys()))

draw_colored_strategy(hard_df, "Strategia podstawowa dla twardych rąk (Hard Hands)", "basic_strategy_hard_hands.png")
draw_colored_strategy(soft_df, "Strategia podstawowa dla miękkich rąk (Soft Hands)", "basic_strategy_soft_hands.png")
draw_colored_strategy(split_df, "Strategia podstawowa dla par (Split Hands)", "basic_strategy_split_hands.png")
