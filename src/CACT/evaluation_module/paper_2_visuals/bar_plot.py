import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,  # use inline math for ticks
    "pgf.rcfonts": False  # don't setup fonts from rc parameters
})
plt.style.use('univie')


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def bar_plot(tags,
             x,
             y,
             hue,
             col,
             aggfunc='mean'
             ):
    # connect to the database
    conn = sqlite3.connect('data/HPC/output.db')
    # if tag is None:
    #     options = pd.read_sql_query("SELECT DISTINCT tag FROM auctions", conn)
    #     print('Select a tag number:')
    #     pprint(options)
    #     tag = input()
    #     tag = options.loc[int(tag), 'tag']
    tags_str = ', '.join([f"'{tag}'" for tag in tags])
    df = pd.read_sql_query(f"SELECT * FROM auctions WHERE tag IN ({tags_str})", conn)
    conn.close()
    print(df)
    # pivot the data
    df_pivot = pd.pivot_table(
        df,
        values=y,
        index=[idx for idx in [x, col] if idx is not None],
        columns=hue,
        aggfunc=aggfunc,
    )
    print(df_pivot)

    # set the size according to the latex article template
    # > 418.25368 pt.
    # > l.25
    plt.figure(figsize=set_size(418.25368, fraction=1, subplots=(1, 1)))

    sns.catplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        col=col,
        kind='bar',
        sharex=True,
        sharey=True,
    )
    plt.show()
    plt.savefig(f'C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home '
                f'Deliveries/05_Texts_and_Publications/Paper 2/images/searborn/{"+".join(tags)}.png')


if __name__ == '__main__':
    bar_plot(
        tags=['NeuralNetwork', 'PartitionBenchmarksRandom+GH'],
        x='fin_auction_num_bidding_jobs',
        y='rel_query_efficiency',
        hue='fin_auction_rr_fitness_function',
        col=None,
        # aggfunc='mean'

    )
