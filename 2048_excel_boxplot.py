import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WEIGHTS_COR = ["emptyw", "edgew", "smoothw", "matchw", "monow", "snakew"]

for weight in WEIGHTS_COR:
    df = pd.read_excel(f'tuning\\{weight}-checking.xlsx', index_col=0)
    desmos_df = boxplot_df = df.drop(['mean', 'best'], axis=1)
    boxplot_df = boxplot_df.T
    boxplot_df.boxplot(column=[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]).set(title=f'{weight}')
    mean_best_df = df[['mean', 'best']].reset_index()

    print(mean_best_df)
    # plt.savefig(f'tuning\\{weight}-boxplot.png')
    plt.show()
    # plt.close()
    #
    # sns.lmplot(x='val', y='mean', data=mean_best_df, fit_reg=True).set(title=f'{weight}')
    # plt.savefig(f'tuning\\{weight}-mean.png')
    # # plt.show()
    # plt.close()
    # sns.lmplot(x='val', y='best', data=mean_best_df, fit_reg=True).set(title=f'{weight}')
    # plt.savefig(f'tuning\\{weight}-best.png')
    # # plt.show()
    # plt.close()
