import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Apply style settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
rcParams['font.size'] = 7

# Parameters
efficiency_snrs_to_plot = (5., 7., 9., 11., 13., 15., 17., 19., 21., 23.)
index_filecol = [1, 4]  # Columns for the specified FAPs
epoch_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 40, 55, 70]

# Filepath pattern
file_pattern = 'Detection/Efficiency_test/src/efficiencies/out_efficiencies_run_0000_epoch_{:04d}.txt'

# Indices for SNR rows in the files corresponding to the specified SNRs
indices_filerow_plot = [0, 1, 2, 4, 6]

# Load data for all epochs and FAPs
data = []  # List to store data for all epochs
for epoch in epoch_axis:
    filepath = file_pattern.format(epoch)
    file_data = np.loadtxt(filepath)  # Load the file
    data.append(file_data)  # Append data to the list

# Convert to a NumPy array for easier processing
data = np.array(data)  # Shape: (n_epochs, n_snrs, n_faps)

# Plot each FAP
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Single row, adjusted aspect ratio
axes = axes.flatten()

for i, (ax, fap_index) in enumerate(zip(axes, index_filecol)):
    # Plot each SNR for the current FAP
    for snr_idx, snr in zip(indices_filerow_plot, efficiency_snrs_to_plot):
        # Extract efficiency values for the current SNR and FAP across epochs
        efficiencies = data[:, snr_idx, fap_index]
        ax.plot(epoch_axis, efficiencies, label=f'SNR={snr:.1f}', linewidth=1)

    # Find the epoch with the highest TAP for the current FAP
    mean_tap = data[:, :, fap_index].mean(axis=1)  # Mean TAP for this FAP across SNRs
    best_epoch_zi = np.argmax(mean_tap)  # Index of the best epoch
    best_epoch = epoch_axis[best_epoch_zi]  # Actual epoch value

    # Add a red dashed line at the best epoch
    ax.axvline(x=best_epoch, color='red', linestyle='dashed', linewidth=1, label='Best Epoch')

    # Customize the plot
    ax.set_title(f'FAP = 1e-{fap_index}', fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(visible=True)

    # Plot legend only for the first subplot
    if i == 0:
        ax.legend(fontsize=6, loc='lower right')
    else:
        ax.legend().set_visible(False)

    # Remove y-axis labels for the second column
    if i == 1:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel('Efficiency', fontsize=8)

    # Add x-axis labels for both plots
    ax.set_xlabel('Epoch', fontsize=8)

# Adjust layout and save the plot
fig.tight_layout()
fig.savefig('Detection/Efficiency_test/src/figures/efficiency_plots_single_row.png')
plt.close(fig)
