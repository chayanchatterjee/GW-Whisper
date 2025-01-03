import matplotlib.pyplot as plt
import pandas as pd
plt.rc('font', family='serif')

# Load the file
file_path = 'efficiencies/out_efficiencies_run_0000_epoch_0055.txt'  # Replace with the actual path to your file

# Read the data
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)

# Extract columns
snr = data[0]
faps = [0.1, 0.01, 0.001, 0.0001]

# Plot the data
plt.figure(figsize=(8, 6))
for i, fap in enumerate(faps, start=1):
    plt.plot(snr, data[i], 'o-', label=f'FAP = {fap}')  # 'o-' for bullets connected by lines

plt.xlim(5,15)
# Customize the plot
plt.xlabel('SNR', fontsize=15)
plt.ylabel('True Alarm Probability', fontsize=15)
#plt.yscale('log')
#plt.title('SNR vs Detection Probability for Various FAPs (Log Scale)')
plt.legend(fontsize=15)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', labelsize=15)
plt.tight_layout()

# Show the plot
plt.savefig('efficiencies/TAP_vs_SNR.png', bbox_inches="tight", facecolor='w', transparent=False, dpi=400)
