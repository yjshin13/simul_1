import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def line_chart(x, title):

    x = pd.DataFrame(x)
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink',
                'tab:olive', 'tab:purple']
    columns = x.columns

    # Draw Plot
    #plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=100)
    # length = np.arange(after_nav.index[0],after_nav.index[-1] + pd.DateOffset(years=1),
    #                         dtype='datetime64[Y]')
    length = np.arange(x.index[0], x.index[-1] + pd.DateOffset(years=1),
                       dtype='datetime64[Y]')
    if len(x.columns) >= 2:

        ax.fill_between(x.index, y1=x.iloc[:, 0].squeeze().values, y2=0, label=columns[0], alpha=0.3,
                        color=mycolors[1], linewidth=2)
        ax.fill_between(x.index, y1=x.iloc[:, 1].squeeze().values, y2=0, label=columns[1], alpha=0.3,
                        color=mycolors[0], linewidth=2)

    else:
        ax.fill_between(x.index, y1=x.squeeze().values, y2=0, label=columns, alpha=0.3, color=mycolors[1],
                        linewidth=2)
    # ax=x.plot(x.index, y1=x.iloc[:, 0].squeeze().values, y2=0, label=columns[0], color=mycolors[1], linewidth=2)
    # ax.set_title('Portfolio NAV', fontsize=18)
    ax.set_xlabel('Time', size=15, labelpad=20)
    ax.set_ylabel('Index', size=15, labelpad=20)
    ax.tick_params(labelsize=16)

    ax.set_xticks(length)
    ax.set_xticklabels(length)
    ax.tick_params(labelsize=16)
    plt.title(title, loc='left', pad=30, size=25)
    plt.xticks(rotation=0)
    plt.legend(loc='upper left')

    plt.ylim(x.min().min() - abs(x.max().max() - x.min().min()) * 0.1,
             x.max().max() + abs(x.max().max() - x.min().min()) * 0.05)

    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    ax.get_legend().remove()

    ax.margins(x=0, y=0)

    #########################[Graph Insert]#####################################

    return ax.figure
