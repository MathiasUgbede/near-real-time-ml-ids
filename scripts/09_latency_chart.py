import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = '/home/mathias/ml-ids-project/results/'
total = np.load(RESULTS_PATH + 'total_times.npy')

plt.figure(figsize=(10, 6))
plt.hist(total, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
plt.axvline(np.median(total), color='red', linestyle='--', linewidth=2,
            label='Median: ' + format(np.median(total), '.1f') + ' ms')
plt.axvline(np.mean(total), color='orange', linestyle='--', linewidth=2,
            label='Mean: ' + format(np.mean(total), '.1f') + ' ms')
plt.xlabel('Detector-Side Processing Latency (ms)')
plt.ylabel('Number of Flows')
plt.title('Two-VM Real-Time Detection Latency Distribution (Random Forest, 990 flows)')
plt.legend()
plt.tight_layout()
out = RESULTS_PATH + 'realtime_latency_histogram.png'
plt.savefig(out, dpi=100, bbox_inches='tight')
print("Chart saved: " + out)
