import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, data_path="..\\diabetes_dataset.csv"):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)

    def plot_heat_map(self):
        plt.figure(figsize=(10, 10))
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap of Diabetes Dataset")
        plt.show()

# Run directly
if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_heat_map()
