import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Weights and Values
WEIGHTS_COR = ["emptyw", "edgew", "smoothw", "matchw", "monow", "snakew"]
VALUES = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]

# Loops through each of the weights
# for weight in WEIGHTS_COR:
if True:
    weight = WEIGHTS_COR[1]
    df = pd.read_excel(f'tuning\\{weight}-checking.xlsx', index_col=0)  # data from excel
    boxplot_df = df.drop(['mean', 'best'], axis=1).T  # transposes table for plot
    boxplot_df.boxplot(column=VALUES).set(title=f'{weight}')  # box plot
    mean_best_df = df[['mean', 'best']].reset_index()  # moves index out of index

    # Loops through to create Desmos list in io
    for val in VALUES:
        print(",\\ ".join(boxplot_df[val].astype(str).tolist()))
        print()

    # Look at table
    print(mean_best_df)

    # Boxplot
    plt.savefig(f'tuning\\{weight}-boxplot.png')
    # plt.show()
    plt.close()

    # Graph of mean with linear regression model
    sns.lmplot(x='val', y='mean', data=mean_best_df, fit_reg=True).set(title=f'{weight}')
    plt.savefig(f'tuning\\{weight}-mean.png')
    # plt.show()
    plt.close()

    # Graph of best with linear regression model
    sns.lmplot(x='val', y='best', data=mean_best_df, fit_reg=True).set(title=f'{weight}')
    plt.savefig(f'tuning\\{weight}-best.png')
    # plt.show()
    plt.close()
