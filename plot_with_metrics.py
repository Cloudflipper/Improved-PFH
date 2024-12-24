
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from adjustText import adjust_text
from sklearn.linear_model import LinearRegression

def plot_data(file_path):
    # Load the Excel file
    df = pd.ExcelFile(file_path).parse('Sheet1')

    # Define colors for different filter functions
    color_mapping = {
        "--": "black",
        "arctan": "red",
        "clip": "green",
        "smooth": "blue"
    }

    # Initialize the plot
    plt.figure(figsize=(12, 8))
    texts = []
    for _, row in df.iterrows():
        x = row["final error"]
        acg = row["average retainment ratio"]
        min_val = row["min"]
        max_val = row["max"]
        lower_bound = row["90min"]
        upper_bound = row["90max"]
        label = f"{row['filter_function']} | {row['retainment_mode']} | {row['clip']}"
        color = color_mapping.get(row["filter_function"], "gray")

        # Plot the values
        plt.scatter(x, acg, color=color, label=label, alpha=0.7, s=50, marker="o", zorder=3)  # ACG
        plt.scatter(x, min_val, color=color, alpha=0.7, s=50, marker="v", zorder=3)  # Min
        plt.scatter(x, max_val, color=color, alpha=0.7, s=50, marker="^", zorder=3)  # Max
        # Plot the confidence interval as vertical lines
        plt.hlines(upper_bound,x-0.0001,x+0.0001,color=color, alpha=1, linestyles="-", lw=2)
        plt.hlines(lower_bound,x-0.0001,x+0.0001,color=color, alpha=1, linestyles="-", lw=2)
        plt.vlines(x, lower_bound, upper_bound, color=color, alpha=1, linestyles="-", lw=2)
        plt.vlines(x, min_val, max_val, color=color, alpha=0.3, linestyles="-", lw=2)
        # Annotate the points
        texts.append(plt.text(x, acg, f"{label}", fontsize=12, ha="right"))
    
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    # Labeling the axes
    plt.xlabel("Final Error", fontsize=16)
    plt.ylabel("Metrics", fontsize=16)
    plt.title("Retainmeng Ratio Metrics v/s Final Error", fontsize=20)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')

    all_x = []
    all_y = []
    for filter_function, group in df.groupby("filter_function"):
        x = group["final error"]
        y = group["average retainment ratio"]

        # Scatter plot for each filter function
        plt.scatter(x, y, color=color_mapping[filter_function], 
            label=f"{filter_function} (Color: {color_mapping[filter_function]})", 
            alpha=0.7, s=50)

        # Perform regression for each group
        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y)
        y_pred = model.predict(x.values.reshape(-1, 1))
        score = model.score(x.values.reshape(-1, 1), y)

        # Plot regression line for each group
        plt.plot(x, y_pred, color=color_mapping[filter_function], linestyle="--", label=f"{filter_function} regression, r^2={score}")

        # Collect data for global regression
        all_x.extend(x.values)
        all_y.extend(y.values)

    # Perform global regression on all data
    # all_x = np.array(all_x)
    # all_y = np.array(all_y)
    # global_model = LinearRegression()
    # global_model.fit(all_x.reshape(-1, 1), all_y)
    # global_y_pred = global_model.predict(all_x.reshape(-1, 1))

    # Plot global regression line
    #plt.plot(all_x, global_y_pred, color="purple", linestyle="-", label="Global regression")

    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("vs.png")

# Replace 'your_file_path.xlsx' with the path to your file
plot_data('filter_test.xlsx')
