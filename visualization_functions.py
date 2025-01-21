import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def histogram_plot(df, x_label, y_label):
    """
    Creates a histogram plot with a kernel density estimate (KDE) overlay.

    Parameters:
    df (DataFrame): The data frame containing the data.
    x_label (str): The column name for the x-axis.
    y_label (str): The column name for the hue (color) dimension.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=df[x_label], hue=y_label, bins=10, kde=True, color='skyblue')
    plt.title('Histogram Plot')
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'Frequency of: {y_label}')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def violin_plot(df, y_label):
    """
    Creates a violin plot to visualize the distribution of a variable.

    Parameters:
    df (DataFrame): The data frame containing the data.
    y_label (str): The column name for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=y_label, data=df, palette='muted', fill=True)
    plt.title('Violin Plot')
    plt.xlabel(f'{y_label}')
    plt.ylabel(f'Distribution of: {y_label}')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def box_plot(df, y_label):
    """
    Creates a box plot to visualize the distribution of a variable.

    Parameters:
    df (DataFrame): The data frame containing the data.
    y_label (str): The column name for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=y_label, data=df, palette='muted')
    plt.title('Box Plot')
    plt.xlabel(f'{y_label}')
    plt.ylabel(f'Distribution of: {y_label}')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def density_plot(df, x_label, y_label):
    """
    Creates a density plot to visualize the distribution of a variable with KDE.

    Parameters:
    df (DataFrame): The data frame containing the data.
    x_label (str): The column name for the x-axis.
    y_label (str): The column name for the hue (color) dimension.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=df[x_label], hue=df[y_label], data=df, palette='muted', fill=True, alpha=0.5, common_norm=False)
    plt.title('Density Plot')
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'Distribution of: {y_label}')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def bar_plot(df, x_label, y_label):
    """
    Creates a bar plot to visualize the relationship between two variables.

    Parameters:
    df (DataFrame): The data frame containing the data.
    x_label (str): The column name for the x-axis.
    y_label (str): The column name for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df[x_label], y=df[y_label], data=df, palette='muted')
    plt.title('Bar Chart')
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'Distribution of: {y_label}')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_feature_trend(df):
    """
    Creates subplots for each feature in the data frame to visualize trends over time.

    Parameters:
    df (DataFrame): The data frame containing the data.
    """
    n_features = len(df.columns)
    n_rows = math.ceil(n_features / 2)

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=df.columns,
        vertical_spacing=0.02,
        horizontal_spacing=0.05
    )

    for i, col_name in enumerate(df.columns):
        row = (i // 2) + 1
        col = (i % 2) + 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col_name], mode='lines', name=col_name),
            row=row, col=col
        )

    fig.update_layout(
        height=450 * n_rows,
        title_text="Subplots for Features (2 per Row)",
        showlegend=False,
        font=dict(size=10),
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12)

    fig.show()

def heat_map_plot(df):
    """
    Creates a heatmap to visualize the correlation matrix of the data frame.

    Parameters:
    df (DataFrame): The data frame containing the data.
    """
    correlation_matrix = df.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        annot_kws={"size": 10},
        xticklabels=correlation_matrix.columns,
        yticklabels=correlation_matrix.columns
    )
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10)
    plt.title("Correlation Heatmap", fontsize=14)
    plt.show()