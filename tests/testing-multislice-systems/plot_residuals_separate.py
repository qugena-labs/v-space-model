import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_residuals_separate(coupled_file, uncoupled_file):
    # Read data from coupled file using pandas to handle duplicates
    df_c = pd.read_csv(coupled_file)
    df_c['Residual'] = df_c['Boundary_Left_Prob'] - df_c['Boundary_Right_Prob']
    df_c = df_c.groupby('Time').mean().reset_index()

    # Read data from uncoupled file
    df_u = pd.read_csv(uncoupled_file)
    df_u['Residual'] = df_u['Boundary_Left_Prob'] - df_u['Boundary_Right_Prob']
    df_u = df_u.groupby('Time').mean().reset_index()

    # Calculate mean absolute residuals
    mean_res_c = np.mean(np.abs(df_c['Residual']))
    mean_res_u = np.mean(np.abs(df_u['Residual']))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

    # Coupled case
    ax1.scatter(df_c['Time'], df_c['Residual'], c='r', label='Coupled (g=100)')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals at the Boundary (Coupled)')
    ax1.legend()
    ax1.grid(True)

    # Uncoupled case
    ax2.scatter(df_u['Time'], df_u['Residual'], c='b', label='Uncoupled (g=0)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residuals at the Boundary (Uncoupled)')
    ax2.legend()
    ax2.grid(True)

    # Add mean residual text
    fig.text(0.5, 0.02, f'Mean Absolute Residual (Coupled): {mean_res_c:.6f}\nMean Absolute Residual (Uncoupled): {mean_res_u:.6f}', ha='center')

    plt.show()

if __name__ == '__main__':
    plot_residuals_separate('coupl_boundary_log.txt', 'ex_nocoupl_boundary_log.txt')