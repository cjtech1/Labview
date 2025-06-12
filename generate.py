import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Time settings
fs = 500  # Sampling frequency (Hz)
duration = 10  # seconds to simulate
t = np.linspace(0, duration, int(fs*duration))

# Define one heartbeat using a simple synthetic ECG waveform
def synthetic_ecg(t, hr=75):
    beat_duration = 60 / hr  # duration of one heartbeat in seconds
    ecg = np.zeros_like(t)

    for beat_start in np.arange(0, t[-1], beat_duration):
        # Time-shifted Gaussian-like waves for P, QRS, and T components
        p_wave = 0.25 * np.exp(-((t - (beat_start + 0.1))**2) / (2 * 0.01**2))
        q_wave = -0.1 * np.exp(-((t - (beat_start + 0.2))**2) / (2 * 0.005**2))
        r_wave = 1.0 * np.exp(-((t - (beat_start + 0.22))**2) / (2 * 0.01**2))
        s_wave = -0.25 * np.exp(-((t - (beat_start + 0.24))**2) / (2 * 0.005**2))
        t_wave = 0.5 * np.exp(-((t - (beat_start + 0.35))**2) / (2 * 0.02**2))
        
        ecg += p_wave + q_wave + r_wave + s_wave + t_wave
    
    return ecg

# Generate synthetic ECG
ecg_signal = synthetic_ecg(t)

# Plot the synthetic ECG
plt.figure(figsize=(15, 4))
plt.plot(t, ecg_signal, label='Synthetic ECG', color='darkblue')
plt.title('Synthetic ECG Signal (First 10 Seconds)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.tight_layout()
plt.legend()
plt.show()


df = pd.DataFrame({'Time (s)': t, 'Voltage (mV)': ecg_signal})
df.to_csv('synthetic_ecg_60s.csv', index=False)

print("ECG data saved to 'synthetic_ecg_60s.csv'")
