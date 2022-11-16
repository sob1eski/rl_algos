#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set_theme(style = 'darkgrid')


def make_df_for_plot(data, model, variable):
    df = pd.DataFrame.from_dict([data[str(rs)][model][variable] for rs in np.arange(0, 8, 1)])
    df = df.stack().droplevel(0)
    df = pd.DataFrame(data = {variable: df.values, 'timestep': df.index, 'model': df.values.shape[0] * [model]})    
    return df

def plot_results(data, variables, models):
    for var in variables:
        dfs = []
        for model in models:
            dfs.append(make_df_for_plot(data, model, var))
        sns.relplot(
            pd.concat(dfs),
            kind = 'line',
            x = 'timestep',
            y = var,
            hue = 'model')
    plt.show()

if __name__ == '__main__':
    path = 'comp_2022-11-16 09:38:54.254033.json'
    with open(path) as f:
        data = json.load(f)
    vars = ['returns', 'ep_lengths', 'losses']
    models = ['spg', 'vpg']
    plot_results(data, vars, models)