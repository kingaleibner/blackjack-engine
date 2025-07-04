import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

log_dir        = "logs"
output_dir     = "analysis_outputs"
rounds_per_run = 500
initial_chips  = 1000

os.makedirs(output_dir, exist_ok=True)
palette = sns.color_palette('Set2')

all_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
frames = []
for file in all_files:
    df = pd.read_csv(os.path.join(log_dir, file))
    df['scenario'] = file.replace(".csv", "")
    frames.append(df)
df_full = pd.concat(frames, ignore_index=True)

df_moves = df_full[df_full['action'].notna()]

last_moves = (
    df_moves
    .sort_values('seq')
    .groupby(['scenario','run_id','player','round'], as_index=False)
    .tail(1)
    [['scenario','run_id','player','round','delta','reward']]
)

last_moves = last_moves.sort_values(
    ['scenario','run_id','player','round']
)
last_moves['cum_chips'] = (
    last_moves
    .groupby(['scenario','run_id','player'])['delta']
    .cumsum()
    + initial_chips
)

df_final = (
    last_moves
    .groupby(['scenario','run_id','player'], as_index=False)
    .agg(final_chips=('cum_chips','last'))
)
df_final.to_csv(os.path.join(output_dir,'final_chips_per_run.csv'), index=False)

metrics = (
    last_moves
    .groupby(['scenario','run_id','player'], as_index=False)
    .agg(total_reward=('delta','sum'))
)
metrics['EV_per_round'] = metrics['total_reward'] / rounds_per_run
metrics['ROI']         = metrics['total_reward'] / initial_chips
metrics.to_csv(os.path.join(output_dir,'metrics_summary.csv'), index=False)

for scen in df_final['scenario'].unique():
    df_f = df_final[df_final['scenario']==scen]
    df_m = metrics[metrics['scenario']==scen]

    plt.figure(figsize=(6,4))
    sns.boxplot(data=df_m, x='player', y='EV_per_round', palette=palette)
    plt.xlabel('Strategia')
    plt.ylabel('EV na rundę')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplot_EV_{scen}.png")
    plt.clf()

    plt.figure(figsize=(6,4))
    sns.histplot(data=df_f, x='final_chips', hue='player',
                 element='step', stat='count', common_norm=False,
                 palette=palette)
    plt.xlabel('Końcowa liczba żetonów podczas gry')
    plt.ylabel('Liczba gier')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hist_final_chips_{scen}.png")
    plt.clf()

    pivot = df_f.pivot(index='run_id', columns='player', values='final_chips')
    plt.figure(figsize=(6,4))
    for i, pl in enumerate(pivot.columns):
        plt.plot(pivot.index, pivot[pl], label=pl, color=palette[i])
    plt.xlabel('Numer symulacji')
    plt.ylabel('Końcowa liczba żetonów')
    plt.legend(title='Strategia')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trend_final_chips_{scen}.png")
    plt.clf()

best = df_final.loc[
    df_final.groupby(['scenario','player'])['final_chips'].idxmax()
]
for _, row in best.iterrows():
    scen, run_id = row['scenario'], row['run_id']
    subset = last_moves[
        (last_moves['scenario']==scen) &
        (last_moves['run_id']==run_id)
    ].sort_values('round')

    plt.figure(figsize=(8,4))
    pal = sns.color_palette("Set2", subset['player'].nunique())
    for (player, grp), c in zip(subset.groupby('player'), pal):
        plt.plot(grp['round'], grp['cum_chips'], label=player, color=c)
    plt.xlabel("Runda")
    plt.ylabel("Żetony")
    plt.legend(title="Gracz", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    fn = f"{output_dir}/chip_progression_best_{scen}_run{run_id}.png"
    plt.savefig(fn, bbox_inches='tight')
    plt.clf()
