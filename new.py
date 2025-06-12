import numpy as np
import pandas as pd

# Function to simulate a single PQRST ECG beat
def simulate_ecg_waveform(num_beats=5, sampling_rate=500, duration_per_beat=1.0):
    t = np.linspace(0, duration_per_beat, int(sampling_rate * duration_per_beat), endpoint=False)
    ecg = (
        0.1 * np.sin(2 * np.pi * 1 * t) +  # P-wave
        -1.5 * np.exp(-((t - 0.3) ** 2) / 0.002) +  # Q-wave
        3.0 * np.exp(-((t - 0.4) ** 2) / 0.001) +  # R-wave
        -0.8 * np.exp(-((t - 0.5) ** 2) / 0.002) +  # S-wave
        0.3 * np.sin(2 * np.pi * 0.5 * (t - 0.6)) * (t > 0.6)  # T-wave
    )

    # Repeat the beat for the specified number of beats
    full_ecg = np.tile(ecg, num_beats)
    total_time = np.linspace(0, num_beats * duration_per_beat, len(full_ecg), endpoint=False)

    return pd.DataFrame({'time': total_time, 'ecg': full_ecg})

# Generate ECG data
ecg_df = simulate_ecg_waveform()

# Save to CSV
ecg_df.to_csv("simulated_pqrst_ecg.csv", index=False)
print("ECG data saved to 'simulated_pqrst_ecg.csv'")
