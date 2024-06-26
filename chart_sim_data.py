import pandas as pd
import matplotlib.pyplot as plt

def plot_from_csv(file_path, x_column, y_column, n, filter_column=None, title=None, x_label=None, y_label=None):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Select the last 'n' rows
    df = df.tail(n)
    
    plt.figure(figsize=(10, 6))
    
    if filter_column:
        unique_values = df[filter_column].unique()
        for value in unique_values:
            filtered_df = df[df[filter_column] == value]
            plt.plot(filtered_df[x_column], filtered_df[y_column], label=value)
        plt.legend(title=filter_column)
    else:
        plt.plot(df[x_column], df[y_column], label=y_column)
    
    # Set labels and title, using provided values if they are not None
    plt.xlabel(x_label if x_label else x_column)
    plt.ylabel(y_label if y_label else y_column)
    plt.title(title if title else f'{y_column} vs {x_column}')
    
    plt.grid(True)
    plt.show()

# Example usage:
n_samples = 20
x_column = 'n_rec'
y_column = 'repayment' #'n_loans' #'profit_divided_by_budget'
filter_column = None
plot_from_csv('lending_simulation_results.csv', x_column, y_column, n_samples, filter_column)
