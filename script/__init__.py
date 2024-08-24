import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SolarDataAnalysis:
    def __init__(self, region_files):
        self.region_files = region_files
        self.data = {region: pd.read_csv(file) for region, file in region_files.items()}

    def clean_data(self, df):
        # Replace negative values with zero
        df[df < 0] = 0
        
        # Use IQR method to remove outliers
        for column in df.columns:
            if df[column].dtype in [np.int64, np.float64]:  # Apply to numerical columns only
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
                
        return df

    def preprocess_data(self):
        for region, df in self.data.items():
            self.data[region] = self.clean_data(df)

    def plot_time_series(self):
        for region, df in self.data.items():
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            plt.figure(figsize=(14, 7))
            plt.subplot(3, 1, 1)
            df[['GHI']].resample('M').mean().plot(ax=plt.gca())
            plt.title(f'{region} - Monthly GHI')
            plt.ylabel('GHI')
            
            plt.subplot(3, 1, 2)
            df[['DNI']].resample('M').mean().plot(ax=plt.gca())
            plt.title(f'{region} - Monthly DNI')
            plt.ylabel('DNI')
            
            plt.subplot(3, 1, 3)
            df[['DHI']].resample('M').mean().plot(ax=plt.gca())
            plt.title(f'{region} - Monthly DHI')
            plt.ylabel('DHI')
            
            plt.tight_layout()
            plt.show()

    def calculate_summary_statistics(self):
        summaries = {}
        for region, df in self.data.items():
            summaries[region] = df.describe().transpose()
            summaries[region].to_csv(f'summary_{region}.csv')
        
        return summaries

    def combine_summaries(self, summaries):
        combined_summary = pd.concat(summaries.values(), axis=1)
        combined_summary.columns = [f'{region}_{col}' for region in summaries.keys() for col in summaries[region].columns]
        combined_summary.to_csv('combined_summary_statistics.csv')
        return combined_summary

    def plot_comparison(self, combined_summary):
        metrics = ['GHI', 'DNI', 'DHI', 'Tamb']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            combined_summary.loc['mean'][[f'{region}_{metric}' for region in self.data.keys()]].plot(kind='bar')
            plt.title(f'Mean {metric} Comparison Across Regions')
            plt.ylabel(f'Mean {metric}')
            plt.show()

