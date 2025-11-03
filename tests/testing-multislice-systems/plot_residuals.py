
import matplotlib.pyplot as plt
import numpy as np

def plot_residuals(coupled_file, uncoupled_file):
    # Read data from coupled file
    time_c, res_c = [], []
    with open(coupled_file, 'r') as f:
        next(f) # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                time_c.append(float(parts[0]))
                res_c.append(float(parts[1]) - float(parts[2]))

    # Read data from uncoupled file
    time_u, res_u = [], []
    with open(uncoupled_file, 'r') as f:
        next(f) # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                time_u.append(float(parts[0]))
                res_u.append(float(parts[1]) - float(parts[2]))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(time_c, res_c, c='r', label='Coupled (g=100)')
    plt.scatter(time_u, res_u, c='b', label='Uncoupled (g=0)')
    plt.xlabel('Time')
    plt.ylabel('Residual (Left Prob - Right Prob)')
    plt.title('Residuals at the Boundary (x=0)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_residuals('coupl_boundary_log.txt', 'ex_nocoupl_boundary_log.txt')
