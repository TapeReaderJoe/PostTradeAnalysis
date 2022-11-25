import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def plot_chart_pta(df: pd.DataFrame, path_for_pdf: str):
    mc = mpf.make_marketcolors(up='#ffffff', down='#000000', edge='black', wick='black', volume={'up': 'gray', 'down': '#000000'}, vcdopcod=False)
    base_style = {'axes.titlesize': 7,
                  'axes.labelsize': 7,
                  'lines.linewidth': 3,
                  'lines.markersize': 4,
                  'ytick.left': False,
                  'ytick.right': True,
                  'ytick.labelleft': False,
                  'ytick.labelright': True,
                  'xtick.labelsize': 6,
                  'ytick.labelsize': 6,
                  'axes.linewidth': 0.8,
                  'savefig.pad_inches': 0.5,
                  'savefig.bbox': 'tight',
                  'grid.alpha': 0.2}

    ibd = mpf.make_mpf_style(marketcolors=mc, mavcolors=['green', 'red', 'black', 'blue'], y_on_right=True,
                             rc=base_style)
    egrid = (21, 29)
    fig = mpf.figure(style=ibd, figsize=(11, 8))
    info_ax = plt.subplot2grid(egrid, (0, 0), colspan=29, rowspan=1)
    price_ax = plt.subplot2grid(egrid, (1, 0), colspan=29, rowspan=16)
    volume_ax = plt.subplot2grid(egrid, (17, 0), colspan=29, rowspan=4, sharex=price_ax)
    volume_ax.tick_params(which='both', labelbottom=False, labeltop=False, labelright=True, bottom=False,
                          top=False, right=False)
    info_ax.tick_params(which='both', labelbottom=False, labeltop=False, labelright=False, bottom=False,
                        top=False, right=False)
    price_ax.tick_params(which='both', labelbottom=False, labeltop=False, labelright=True, bottom=False,
                         top=False, right=False)

    stock_name = df['name'].values[-1]
    vol50 = mpf.make_addplot(df['volume_sma50'], color='red', width=0.6, ax=volume_ax)
    sma10 = mpf.make_addplot(
        df['sma10'], ax=price_ax, color='red', width=0.6)
    sma20 = mpf.make_addplot(
        df['sma20'], ax=price_ax, color='blue', width=0.6)
    sma50 = mpf.make_addplot(df['sma50'], ax=price_ax, color='green', width=0.6)
    buys = mpf.make_addplot(df['buy_signals'].values, type='scatter', ax=price_ax, color='green', markersize=50, marker=r'$\Uparrow$')
    sells = mpf.make_addplot(df['sell_signals'].values, type='scatter', ax=price_ax, color='red', markersize=50, marker=r'$\Downarrow$')

    kwargs = dict(horizontalalignment='center', color='#000000', fontsize=5, backgroundcolor='white',
                  bbox=dict(boxstyle='square', fc='white', ec='none', pad=0))
    window_volume = 10
    for i in range(window_volume, len(df)):
        df_slice = df[i-window_volume: i+window_volume].copy()
        if df['rvol'].iloc[i] > 1.5 and df_slice['rvol'].max() <= df['rvol'].iloc[i]:
            volume_ax.text(i + 1, df['volume'].iloc[i]*1.0, str(int(df['rvol'].iloc[i]*100)) + "%\n" + str(np.round(df['volume'].iloc[i]/1e6, 1)) +"M", **kwargs,
                           verticalalignment='bottom')
    mpf.plot(df[['open', 'high', 'low', 'close', 'volume']], type='candle', ax=price_ax, volume=volume_ax, addplot=[sma10, sma20, sma50, vol50, buys, sells],
             datetime_format="%b'%y", xrotation=0, tight_layout=False,
             scale_width_adjustment=dict(volume=0.9),
             update_width_config=dict(ohlc_ticksize=1))

    info_ax.text(0.01, 0.2, f"{stock_name}", fontsize=10)
    info_ax.text(0.8, 0.2, f"{df.index[0].strftime('%Y-%m-%d')} - {df.index[-1].strftime('%Y-%m-%d')}", fontsize=8)
    volume_ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: str(np.round(x / 1000000, 1)) + 'M'))
    price_ax.grid(False)
    volume_ax.grid(False)
    price_ax.xaxis.set_major_locator(mticker.AutoLocator())
    price_ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    price_ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    volume_ax.set_ylabel("Volume")
    volume_ax.yaxis.set_label_position("right")
    volume_ax.legend(['50d avg'], loc="upper left", framealpha=1, fontsize=6)
    price_ax.margins(x=0.01, y=0.1)
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(path_for_pdf)
    print(f'Saved pdf {stock_name}')
    plt.close('all')



