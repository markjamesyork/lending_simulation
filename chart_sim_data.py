import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import product

def plot_from_csv(file_path, x_column, y_column, n, filter_columns=None, title=None, x_label=None, y_label=None):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Select the last 'n' rows
    df = df.tail(n)
    
    plt.figure(figsize=(10, 6))
    
    if filter_columns:
        unique_combinations = df[filter_columns].drop_duplicates()
        for combination in unique_combinations.itertuples(index=False):
            filter_condition = (df[filter_columns] == combination).all(axis=1)
            filtered_df = df[filter_condition]
            label = ', '.join([f"{col}={getattr(combination, col)}" for col in filter_columns])
            plt.plot(filtered_df[x_column], filtered_df[y_column], label=label)
        plt.legend(title=', '.join(filter_columns))
    else:
        plt.plot(df[x_column], df[y_column], label=y_column)
    
    # Set labels and title, using provided values if they are not None
    x_label_text = x_label if x_label else x_column
    y_label_text = y_label if y_label else y_column
    title_text = title if title else f'{y_column} vs {x_column}'
    
    plt.xlabel(x_label_text)
    plt.ylabel(y_label_text)
    plt.title(title_text)
    
    plt.grid(True)
    
    # Save the plot as a .png file in the 'charts' directory
    if not os.path.exists('charts'):
        os.makedirs('charts')
    
    # Replace spaces with underscores and make the title filename-friendly
    filename = title_text.replace(' ', '_') + '.png'
    plt.savefig(os.path.join('charts', filename))

    plt.show()

# Example usage:
n_samples = 40
x_column = 'n_rec'
y_column = 'profit_divided_by_budget'
filter_columns = ['mean_borrower_rpmt', 'another_filter_column'] # Add the list of filter columns
plot_from_csv('lending_simulation_results.csv', x_column, y_column, n_samples, filter_columns, title="Profit vs Recommendations", x_label="Number of Recommendations", y_label="Profit Divided by Budget")
