# Advanced ECG Visualization Application

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import time
import os

# Try to import scipy, but provide fallback if not available
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class ECGApp:
    def __init__(self, master):
        self.master = master
        self.master.title("ECG Visualization Suite")
        self.master.geometry("1000x700")
        self.master.configure(bg='#2E2E2E')
        
        # Set theme colors
        self.bg_color = '#2E2E2E'
        self.accent_color = '#4CAF50'
        self.text_color = '#FFFFFF'
        self.plot_color = '#00FF00'  # Bright green like medical monitors
        
        # Data variables
        self.canvas = None
        self.animation = None
        self.time_data = []
        self.ecg_data = []
        self.data_file = "synthetic_ecg_60s.csv"
        self.window_size = 5  # Show 5 seconds of data at a time
        self.is_playing = False
        self.current_index = 0
        self.update_speed = 50  # milliseconds between updates (20 fps)
        self.data_increment = 5  # Points to add per update
        self.heart_rate = 0
        
        # Layout design
        self.setup_layout()
        
        # Load initial data
        self.load_data(self.data_file)
        
        # Initialize the plot
        self.setup_plot()

    def setup_layout(self):
        # Create main frames
        self.control_frame = tk.Frame(self.master, bg=self.bg_color, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.right_panel = tk.Frame(self.master, bg=self.bg_color)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.plot_frame = tk.Frame(self.right_panel, bg=self.bg_color)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.stats_frame = tk.Frame(self.right_panel, bg=self.bg_color, height=100)
        self.stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Set up styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", 
                        font=("Segoe UI", 10, "bold"),
                        padding=8,
                        background=self.accent_color,
                        foreground=self.text_color)
        style.map("TButton",
                  background=[("active", "#45a049"), ("disabled", "#555555")])
        style.configure("TScale", 
                        background=self.bg_color, 
                        troughcolor="#555555")
        
        # Control panel elements
        title_label = tk.Label(self.control_frame, text="ECG Controls", 
                               font=("Segoe UI", 14, "bold"), 
                               bg=self.bg_color, fg=self.text_color)
        title_label.pack(pady=(0, 20))
        
        # File selection
        file_frame = tk.Frame(self.control_frame, bg=self.bg_color)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        file_btn = ttk.Button(file_frame, text="Load ECG File", 
                              command=self.browse_file)
        file_btn.pack(fill=tk.X)
        
        # Playback controls
        btn_frame = tk.Frame(self.control_frame, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.play_btn = ttk.Button(btn_frame, text="▶ Play", 
                                  command=self.toggle_playback)
        self.play_btn.pack(fill=tk.X, pady=5)
        
        self.reset_btn = ttk.Button(btn_frame, text="⟲ Reset", 
                                   command=self.reset_simulation)
        self.reset_btn.pack(fill=tk.X, pady=5)
        
        # Speed control
        speed_frame = tk.Frame(self.control_frame, bg=self.bg_color)
        speed_frame.pack(fill=tk.X, pady=(0, 15))
        
        speed_label = tk.Label(speed_frame, text="Display Speed:",
                               font=("Segoe UI", 10), 
                               bg=self.bg_color, fg=self.text_color)
        speed_label.pack(anchor=tk.W)
        
        self.speed_scale = ttk.Scale(speed_frame, from_=1, to=20, 
                                    orient=tk.HORIZONTAL, 
                                    command=self.update_speed_value)
        self.speed_scale.set(10)  # Default speed
        self.speed_scale.pack(fill=tk.X, pady=5)
        
        self.speed_value_label = tk.Label(speed_frame, text="1.0x",
                                         font=("Segoe UI", 10), 
                                         bg=self.bg_color, fg=self.text_color)
        self.speed_value_label.pack(anchor=tk.E)
        
        # Window size control
        window_frame = tk.Frame(self.control_frame, bg=self.bg_color)
        window_frame.pack(fill=tk.X, pady=(0, 15))
        
        window_label = tk.Label(window_frame, text="Display Window (seconds):",
                               font=("Segoe UI", 10), 
                               bg=self.bg_color, fg=self.text_color)
        window_label.pack(anchor=tk.W)
        
        self.window_scale = ttk.Scale(window_frame, from_=1, to=10, 
                                     orient=tk.HORIZONTAL, 
                                     command=self.update_window_size)
        self.window_scale.set(5)  # Default window size
        self.window_scale.pack(fill=tk.X, pady=5)
        
        self.window_value_label = tk.Label(window_frame, text="5 sec",
                                          font=("Segoe UI", 10), 
                                          bg=self.bg_color, fg=self.text_color)
        self.window_value_label.pack(anchor=tk.E)
        
        # Statistics display
        self.hr_display = tk.Label(self.stats_frame, 
                                  text="Heart Rate: -- BPM", 
                                  font=("Segoe UI", 14, "bold"), 
                                  bg=self.bg_color, fg="#FF9800")
        self.hr_display.pack(side=tk.LEFT, padx=20)
        
        self.time_display = tk.Label(self.stats_frame, 
                                    text="Time: 0.00s / 0.00s", 
                                    font=("Segoe UI", 14), 
                                    bg=self.bg_color, fg=self.text_color)
        self.time_display.pack(side=tk.RIGHT, padx=20)

    def load_data(self, filename):
        try:
            self.df = pd.read_csv(filename)
            self.time_data = []
            self.ecg_data = []
            self.current_index = 0
            
            # Get column names for flexibility
            time_col = self.df.columns[0]
            ecg_col = self.df.columns[1]
            
            # Get total duration for display
            self.total_duration = self.df[time_col].iloc[-1]
            
            # Update time display
            self.time_display.config(text=f"Time: 0.00s / {self.total_duration:.2f}s")
            
            # Reset plotting 
            if hasattr(self, 'line'):
                self.setup_plot()
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data file: {str(e)}")
            return False

    def browse_file(self):
        """Allow user to select a CSV file"""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select ECG Data File",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        
        if filename:
            self.data_file = filename
            success = self.load_data(filename)
            if success:
                self.reset_simulation()

    def setup_plot(self):
        """Initialize the plot once"""
        # Clear previous plot
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 5), dpi=100)
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # Create empty line
        self.line, = self.ax.plot([], [], color=self.plot_color, linewidth=1.5)
        
        # Style the plot
        self.ax.set_title("ECG Signal", color='white', fontsize=14)
        self.ax.set_xlabel("Time (s)", color='white')
        self.ax.set_ylabel("Voltage (mV)", color='white')
        self.ax.tick_params(colors='white')
        self.ax.grid(True, color='#333333', linestyle='-', alpha=0.7)
        
        # Add grid lines that look like an ECG paper
        self.ax.grid(True, which='minor', color='#333333', linestyle='-', alpha=0.3)
        self.ax.minorticks_on()
        
        # Style the spines
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # Add tight layout
        self.fig.tight_layout()
        
    def update_plot(self):
        """Update the plot with new data points"""
        if not self.is_playing or self.current_index >= len(self.df):
            return
        
        # Add new data points
        for _ in range(min(self.data_increment, len(self.df) - self.current_index)):
            self.time_data.append(self.df.iloc[self.current_index, 0])
            self.ecg_data.append(self.df.iloc[self.current_index, 1])
            self.current_index += 1
        
        # If we have enough data, calculate heart rate
        if len(self.ecg_data) > 100:
            self.calculate_heart_rate()
        
        # Update the plot
        if len(self.time_data) > 0:
            # Determine the visible time window
            current_time = self.time_data[-1]
            start_time = max(0, current_time - self.window_size)
            
            # Get the visible data window
            visible_indices = [i for i, t in enumerate(self.time_data) if start_time <= t <= current_time]
            
            if visible_indices:
                visible_time = [self.time_data[i] for i in visible_indices]
                visible_ecg = [self.ecg_data[i] for i in visible_indices]
                
                # Update the line data
                self.line.set_data(visible_time, visible_ecg)
                
                # Adjust the plot limits
                self.ax.set_xlim(start_time, current_time)
                
                # Set y limits with some padding
                if visible_ecg:
                    min_val = min(visible_ecg)
                    max_val = max(visible_ecg)
                    padding = (max_val - min_val) * 0.1
                    self.ax.set_ylim(min_val - padding, max_val + padding)
                
                # Update the time display
                self.time_display.config(text=f"Time: {current_time:.2f}s / {self.total_duration:.2f}s")
                
                # Draw the plot
                self.canvas.draw_idle()  # This is faster than full redraw
        
        # Schedule next update
        if self.current_index < len(self.df):
            self.master.after(self.update_speed, self.update_plot)
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")

    def calculate_heart_rate(self):
        """Calculate and display heart rate from the ECG data"""
        try:
            # Use the last 5 seconds of data for calculation
            window_size = 5  # seconds
            sampling_rate = 500  # Estimate from data density
            
            # Get the last section of data
            recent_data = self.ecg_data[-int(min(window_size * sampling_rate, len(self.ecg_data))):]
            
            if SCIPY_AVAILABLE:
                # Find R peaks (highest points in ECG)
                peaks, _ = find_peaks(recent_data, height=0.5, distance=sampling_rate*0.4)
                
                if len(peaks) > 1:
                    # Calculate heart rate
                    avg_interval = np.mean(np.diff(peaks)) / sampling_rate  # in seconds
                    self.heart_rate = int(60 / avg_interval)
                    
                    # Update the display
                    self.hr_display.config(text=f"Heart Rate: {self.heart_rate} BPM")
                else:
                    self.hr_display.config(text="Heart Rate: -- BPM")
            else:
                # Simple fallback method if scipy is not available
                # Use basic threshold detection
                threshold = np.mean(recent_data) + np.std(recent_data)
                above_threshold = np.where(np.array(recent_data) > threshold)[0]
                
                # Find sequences of consecutive points (peaks)
                peaks = []
                if len(above_threshold) > 0:
                    # Group consecutive indices
                    groups = np.split(above_threshold, np.where(np.diff(above_threshold) != 1)[0] + 1)
                    
                    # Get the max value index in each group as the peak
                    for group in groups:
                        if len(group) > 0:
                            peak_idx = group[np.argmax([recent_data[i] for i in group])]
                            peaks.append(peak_idx)
                
                if len(peaks) > 1:
                    # Calculate average interval between peaks
                    avg_interval = np.mean(np.diff(peaks)) / sampling_rate  # in seconds
                    self.heart_rate = int(60 / avg_interval)
                    
                    # Update the display
                    self.hr_display.config(text=f"Heart Rate: {self.heart_rate} BPM")
                else:
                    self.hr_display.config(text="Heart Rate: -- BPM")
                
        except Exception as e:
            print(f"Error calculating heart rate: {e}")
            self.hr_display.config(text="Heart Rate: -- BPM")
    
    def toggle_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pause")
            
            # If at the end, restart
            if self.current_index >= len(self.df):
                self.reset_simulation()
            
            self.update_plot()
    
    def reset_simulation(self):
        """Reset the simulation to the beginning"""
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        self.current_index = 0
        self.time_data = []
        self.ecg_data = []
        self.time_display.config(text=f"Time: 0.00s / {self.total_duration:.2f}s")
        self.hr_display.config(text="Heart Rate: -- BPM")
        
        # Reset the plot
        self.line.set_data([], [])
        self.canvas.draw()
    
    def update_speed_value(self, value):
        """Update playback speed based on slider value"""
        speed_factor = float(value)
        # Convert to a reasonable speed range (20ms to 200ms)
        self.update_speed = int(200 / speed_factor)
        self.data_increment = max(1, int(speed_factor))
        
        # Update the label
        self.speed_value_label.config(text=f"{speed_factor:.1f}x")
    
    def update_window_size(self, value):
        """Update the time window size displayed"""
        self.window_size = float(value)
        self.window_value_label.config(text=f"{self.window_size:.1f} sec")
        
        # If we have data, update the plot
        if hasattr(self, 'line') and len(self.time_data) > 0:
            # Update the plot with new window size
            current_time = self.time_data[-1]
            start_time = max(0, current_time - self.window_size)
            self.ax.set_xlim(start_time, current_time)
            self.canvas.draw_idle()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()
