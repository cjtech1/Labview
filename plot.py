# Advanced ECG Visualization Application with Clinical Standard Format

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import time
import os

# Try to import scipy, but provide fallback if not available
try:
    from scipy.signal import find_peaks, butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class ECGApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Standard ECG Monitoring System")
        self.master.geometry("1100x750")
        self.master.configure(bg="#F0F0F0")  # Light gray background like medical devices
        
        # Add resize event binding to maintain aspect ratio
        self.master.bind("<Configure>", self.on_resize)
        
        # Set theme colors for standard medical monitor
        self.bg_color = '#F0F0F0'  # Light gray background
        self.accent_color = '#346FBF'  # Medical blue
        self.text_color = '#000000'  # Black text
        self.plot_color = '#000000'  # Standard ECG is black on ECG paper
        
        # Data variables
        self.canvas = None
        self.animation = None
        self.time_data = []
        self.ecg_data = []
        self.filtered_ecg_data = []
        self.data_file = "synthetic_ecg_60s.csv"
        self.window_size = 5  # Show 5 seconds of data at a time (typical ECG strip)
        self.is_playing = False
        self.current_index = 0
        self.update_speed = 40  # milliseconds between updates (25 fps)
        self.data_increment = 6  # Points to add per update
        self.heart_rate = 0
        
        # Standard ECG settings
        self.paper_speed = 25  # mm/sec (standard is 25mm/sec)
        self.gain = 10  # mm/mV (standard is 10mm/mV)
        self.show_grid = True
        self.show_labels = True
        self.calibration_pulse = True
        self.standard_y_min = -0.5  # mV
        self.standard_y_max = 1.5   # mV
        
        # Layout design
        self.setup_layout()
        
        # Load initial data
        self.load_data(self.data_file)
        
        # Initialize the plot with standard ECG format
        self.setup_plot()

    def setup_layout(self):
        # Create main frames
        self.control_frame = tk.Frame(self.master, bg=self.bg_color, padx=10, pady=10, width=220)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.control_frame.pack_propagate(False)  # Prevent frame from shrinking
        
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
                        font=("Arial", 10),
                        padding=8,
                        background=self.accent_color,
                        foreground=self.text_color)
        style.map("TButton",
                  background=[("active", "#2A569F"), ("disabled", "#CCCCCC")])
        style.configure("TScale", 
                        background=self.bg_color, 
                        troughcolor="#DDDDDD")
        style.configure("TCheckbutton",
                        background=self.bg_color)
        style.configure("TCombobox",
                        padding=5)
        
        # Title and header
        header_frame = tk.Frame(self.control_frame, bg=self.bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="ECG Configuration", 
                               font=("Arial", 12, "bold"), 
                               bg=self.bg_color, fg="#003366")
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame, text="Standard 12-Lead Setup", 
                                 font=("Arial", 9), 
                                 bg=self.bg_color, fg="#666666")
        subtitle_label.pack()
        
        # Separator
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # File selection
        file_frame = tk.Frame(self.control_frame, bg=self.bg_color)
        file_frame.pack(fill=tk.X, pady=(5, 10))
        
        file_btn = ttk.Button(file_frame, text="Load ECG Recording", 
                              command=self.browse_file)
        file_btn.pack(fill=tk.X)
        
        # ECG Settings Frame
        settings_frame = tk.LabelFrame(self.control_frame, text="ECG Settings", 
                                      bg=self.bg_color, fg="#003366",
                                      font=("Arial", 10, "bold"),
                                      padx=5, pady=5)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Paper speed control
        speed_frame = tk.Frame(settings_frame, bg=self.bg_color)
        speed_frame.pack(fill=tk.X, pady=5)
        
        speed_label = tk.Label(speed_frame, text="Paper Speed:",
                              font=("Arial", 9), 
                              bg=self.bg_color, fg=self.text_color)
        speed_label.pack(side=tk.LEFT)
        
        self.speed_var = tk.StringVar(value="25 mm/s")
        speed_options = ["12.5 mm/s", "25 mm/s", "50 mm/s"]
        self.speed_menu = ttk.Combobox(speed_frame, 
                                      textvariable=self.speed_var,
                                      values=speed_options,
                                      width=8,
                                      state="readonly")
        self.speed_menu.pack(side=tk.RIGHT)
        self.speed_menu.bind("<<ComboboxSelected>>", self.update_ecg_settings)
        
        # Gain/Sensitivity control
        gain_frame = tk.Frame(settings_frame, bg=self.bg_color)
        gain_frame.pack(fill=tk.X, pady=5)
        
        gain_label = tk.Label(gain_frame, text="Gain:",
                             font=("Arial", 9), 
                             bg=self.bg_color, fg=self.text_color)
        gain_label.pack(side=tk.LEFT)
        
        self.gain_var = tk.StringVar(value="10 mm/mV")
        gain_options = ["5 mm/mV", "10 mm/mV", "20 mm/mV"]
        self.gain_menu = ttk.Combobox(gain_frame, 
                                     textvariable=self.gain_var,
                                     values=gain_options,
                                     width=8,
                                     state="readonly")
        self.gain_menu.pack(side=tk.RIGHT)
        self.gain_menu.bind("<<ComboboxSelected>>", self.update_ecg_settings)
        
        # Lead selection
        lead_frame = tk.Frame(settings_frame, bg=self.bg_color)
        lead_frame.pack(fill=tk.X, pady=5)
        
        lead_label = tk.Label(lead_frame, text="Lead:",
                             font=("Arial", 9), 
                             bg=self.bg_color, fg=self.text_color)
        lead_label.pack(side=tk.LEFT)
        
        self.lead_var = tk.StringVar(value="II")
        lead_options = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.lead_menu = ttk.Combobox(lead_frame, 
                                     textvariable=self.lead_var,
                                     values=lead_options,
                                     width=8,
                                     state="readonly")
        self.lead_menu.pack(side=tk.RIGHT)
        self.lead_menu.bind("<<ComboboxSelected>>", self.update_lead)
        
        # Display options with checkboxes
        options_frame = tk.Frame(settings_frame, bg=self.bg_color)
        options_frame.pack(fill=tk.X, pady=5)
        
        # Grid option
        self.grid_var = tk.BooleanVar(value=True)
        grid_check = ttk.Checkbutton(options_frame, 
                                    text="Show Grid",
                                    variable=self.grid_var, 
                                    command=self.toggle_grid)
        grid_check.pack(anchor=tk.W)
        
        # Wave labels option
        self.label_var = tk.BooleanVar(value=True)
        label_check = ttk.Checkbutton(options_frame, 
                                     text="Show Wave Labels",
                                     variable=self.label_var,
                                     command=self.toggle_labels)
        label_check.pack(anchor=tk.W)
        
        # Calibration pulse option
        self.cal_var = tk.BooleanVar(value=True)
        cal_check = ttk.Checkbutton(options_frame, 
                                   text="Show Calibration",
                                   variable=self.cal_var,
                                   command=self.toggle_calibration)
        cal_check.pack(anchor=tk.W)
        
        # Separator
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Playback controls
        playback_frame = tk.LabelFrame(self.control_frame, text="Recording Controls", 
                                      bg=self.bg_color, fg="#003366",
                                      font=("Arial", 10, "bold"),
                                      padx=5, pady=5)
        playback_frame.pack(fill=tk.X, pady=5)
        
        btn_frame = tk.Frame(playback_frame, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ Play", 
                                  command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.reset_btn = ttk.Button(btn_frame, text="⟲ Reset", 
                                   command=self.reset_simulation)
        self.reset_btn.pack(side=tk.RIGHT, padx=2, fill=tk.X, expand=True)
        
        # Speed control
        playback_speed_frame = tk.Frame(playback_frame, bg=self.bg_color)
        playback_speed_frame.pack(fill=tk.X, pady=5)
        
        playback_speed_label = tk.Label(playback_speed_frame, text="Playback Speed:",
                                       font=("Arial", 9), 
                                       bg=self.bg_color, fg=self.text_color)
        playback_speed_label.pack(anchor=tk.W)
        
        self.speed_scale = ttk.Scale(playback_speed_frame, from_=1, to=20, 
                                    orient=tk.HORIZONTAL, 
                                    command=self.update_speed_value)
        self.speed_scale.set(10)  # Default speed
        self.speed_scale.pack(fill=tk.X, pady=2)
        
        speeds_label_frame = tk.Frame(playback_speed_frame, bg=self.bg_color)
        speeds_label_frame.pack(fill=tk.X)
        
        tk.Label(speeds_label_frame, text="Slow", font=("Arial", 7),
                bg=self.bg_color, fg=self.text_color).pack(side=tk.LEFT)
        
        self.speed_value_label = tk.Label(speeds_label_frame, text="1.0x",
                                         font=("Arial", 8, "bold"), 
                                         bg=self.bg_color, fg=self.text_color)
        self.speed_value_label.pack(side=tk.LEFT, expand=True)
        
        tk.Label(speeds_label_frame, text="Fast", font=("Arial", 7),
                bg=self.bg_color, fg=self.text_color).pack(side=tk.RIGHT)
        
        # Window size control
        window_frame = tk.Frame(playback_frame, bg=self.bg_color)
        window_frame.pack(fill=tk.X, pady=5)
        
        window_label = tk.Label(window_frame, text="Time Window:",
                               font=("Arial", 9), 
                               bg=self.bg_color, fg=self.text_color)
        window_label.pack(anchor=tk.W)
        
        self.window_scale = ttk.Scale(window_frame, from_=2, to=10, 
                                     orient=tk.HORIZONTAL, 
                                     command=self.update_window_size)
        self.window_scale.set(6)  # Default window size
        self.window_scale.pack(fill=tk.X, pady=2)
        
        windows_label_frame = tk.Frame(window_frame, bg=self.bg_color)
        windows_label_frame.pack(fill=tk.X)
        
        tk.Label(windows_label_frame, text="2 sec", font=("Arial", 7),
                bg=self.bg_color, fg=self.text_color).pack(side=tk.LEFT)
        
        self.window_value_label = tk.Label(windows_label_frame, text="6 sec",
                                          font=("Arial", 8, "bold"), 
                                          bg=self.bg_color, fg=self.text_color)
        self.window_value_label.pack(side=tk.LEFT, expand=True)
        
        tk.Label(windows_label_frame, text="10 sec", font=("Arial", 7),
                bg=self.bg_color, fg=self.text_color).pack(side=tk.RIGHT)
        
        # Statistics panel in a professional format
        # Vitals display
        vitals_frame = tk.LabelFrame(self.stats_frame, text="Patient Vitals", 
                                    bg=self.bg_color, fg="#003366",
                                    font=("Arial", 10, "bold"),
                                    padx=10, pady=5)
        vitals_frame.pack(side=tk.LEFT, padx=10)
        
        # Heart rate with colored background to mimic medical monitors
        hr_container = tk.Frame(vitals_frame, bg='#000000', bd=1, relief=tk.RIDGE, padx=1, pady=1)
        hr_container.pack(pady=5)
        
        hr_title = tk.Label(hr_container, text="HR", font=("Arial", 10), 
                           bg='#000000', fg='#FFFFFF')
        hr_title.pack()
        
        self.hr_display = tk.Label(hr_container, text="--", 
                                  font=("Arial", 20, "bold"), 
                                  bg='#000000', fg='#00FF00',  # Green like medical monitors
                                  width=4)
        self.hr_display.pack()
        
        hr_unit = tk.Label(hr_container, text="BPM", font=("Arial", 8), 
                          bg='#000000', fg='#FFFFFF')
        hr_unit.pack()
        
        # Time display and status
        status_frame = tk.Frame(self.stats_frame, bg=self.bg_color)
        status_frame.pack(side=tk.RIGHT, padx=10)
        
        self.status_label = tk.Label(status_frame, text="Status: Ready", 
                                    font=("Arial", 10), 
                                    bg=self.bg_color, fg="#003366")
        self.status_label.pack(anchor=tk.E)
        
        self.time_display = tk.Label(status_frame, 
                                    text="Time: 0.00s / 0.00s", 
                                    font=("Arial", 10), 
                                    bg=self.bg_color, fg=self.text_color)
        self.time_display.pack(anchor=tk.E)
        
    def toggle_grid(self):
        """Toggle ECG grid visibility"""
        self.show_grid = self.grid_var.get()
        
        if hasattr(self, 'ax'):
            if self.show_grid:
                # First the minor grid (1mm x 1mm squares in pink) - 0.04s x 0.1mV
                self.ax.grid(True, which='minor', color='#FF9999', linestyle='-', linewidth=0.5, alpha=0.8)
                
                # Then the major grid (5mm x 5mm squares in red) - 0.2s x 0.5mV (draw on top of minor)
                self.ax.grid(True, which='major', color='#FF0000', linestyle='-', linewidth=1.0, alpha=0.9)
                
                # Show time markers
                if hasattr(self, 'time_markers'):
                    for marker in self.time_markers:
                        marker.set_visible(True)
                
                # Make the baseline (0 mV) stand out
                if hasattr(self, 'baseline'):
                    self.baseline.set_visible(True)
            else:
                # Hide all grid lines
                self.ax.grid(False, which='both')
                
                # Hide time markers too
                if hasattr(self, 'time_markers'):
                    for marker in self.time_markers:
                        marker.set_visible(False)
                
                # Keep only the baseline if we have it
                if hasattr(self, 'baseline'):
                    self.baseline.set_visible(True)
                
            self.canvas.draw_idle()
        
    def toggle_labels(self):
        """Toggle ECG wave component labels visibility"""
        self.show_labels = self.label_var.get()
        
        if hasattr(self, 'labels'):
            if not self.show_labels:
                # Remove all wave labels
                for wave in self.labels.values():
                    if wave['text'] is not None:
                        wave['text'].remove()
                        wave['text'] = None
                if hasattr(self, 'interval_text'):
                    self.interval_text.set_text("")
            self.canvas.draw_idle()
    
    def toggle_calibration(self):
        """Toggle calibration pulse visibility"""
        self.calibration_pulse = self.cal_var.get()
        
        if hasattr(self, 'cal_rect'):
            self.cal_rect.set_visible(self.calibration_pulse)
            self.canvas.draw_idle()
    
    def update_ecg_settings(self, event=None):
        """Update ECG paper speed and gain settings"""
        # Get paper speed
        speed_str = self.speed_var.get()
        if "12.5" in speed_str:
            self.paper_speed = 12.5
        elif "50" in speed_str:
            self.paper_speed = 50
        else:
            self.paper_speed = 25  # default
        
        # Get gain/sensitivity
        gain_str = self.gain_var.get()
        if "5" in gain_str:
            self.gain = 5
        elif "20" in gain_str:
            self.gain = 20
        else:
            self.gain = 10  # default
        
        # Update the plot with new settings
        if hasattr(self, 'ax'):
            # Update x-axis based on paper speed
            self.ax.xaxis.set_major_locator(MultipleLocator(0.2 * (25 / self.paper_speed)))
            self.ax.xaxis.set_minor_locator(MultipleLocator(0.04 * (25 / self.paper_speed)))
            
            # Update y-axis based on gain
            self.ax.yaxis.set_major_locator(MultipleLocator(0.5 * (10 / self.gain)))
            self.ax.yaxis.set_minor_locator(MultipleLocator(0.1 * (10 / self.gain)))
            
            # Adjust aspect ratio to maintain perfect squares based on new settings
            # Standard is 25 mm/sec and 10 mm/mV, so we adjust relative to these values
            # The aspect ratio should make the grid perfectly square
            new_aspect = (0.2 * (25 / self.paper_speed)) / (0.5 * (10 / self.gain))
            self.ax.set_aspect(new_aspect)
            
            # Update calibration pulse
            if hasattr(self, 'cal_rect'):
                # Calibration pulse is 1mV x 100ms at standard settings
                width = 0.1 * (25 / self.paper_speed)  # Adjust width based on paper speed
                height = 1.0 * (self.gain / 10)  # Adjust height based on gain
                self.cal_rect.set_width(width)
                self.cal_rect.set_height(height)
                
                # Update position to keep it in view
                x_pos = min(0.05, self.ax.get_xlim()[0] + 0.05)
                self.cal_rect.set_xy((x_pos, -0.2))
                
                # Update calibration text positions
                if hasattr(self, 'cal_text'):
                    self.cal_text.set_position((x_pos, -0.3))
                if hasattr(self, 'time_scale_text'):
                    self.time_scale_text.set_text(f"{self.paper_speed} mm/sec")
                    self.time_scale_text.set_position((x_pos + 0.2, -0.4))
            
            # Redraw the plot with updated settings
            self.canvas.draw_idle()
            
    def update_lead(self, event=None):
        """Update the lead display"""
        lead = self.lead_var.get()
        
        if hasattr(self, 'lead_label'):
            self.lead_label.set_text(f"Lead {lead}")
            # In a real application, this would switch to different lead data
            self.canvas.draw_idle()

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
        """Initialize the plot with standard ECG paper format"""
        # Clear previous plot
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Create figure with standard ECG paper format
        # Use a fixed aspect ratio to ensure perfect squares in the grid
        self.fig = plt.figure(figsize=(10, 5), dpi=100)
        
        # Standard ECG paper is typically pale pink/salmon color
        # Using the exact shade found in standard ECG papers
        self.fig.patch.set_facecolor('#FFECD9')  # Standardized ECG paper color
        
        # Create gridspec for main ECG plot and calibration box
        gs = gridspec.GridSpec(1, 1)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax.set_facecolor('#FFECD9')  # Standardized ECG paper color
        
        # Create empty line for the ECG trace
        self.line, = self.ax.plot([], [], color='#000000', linewidth=1.2)  # Standard black trace
        
        # Set aspect ratio to 'equal' to ensure perfect squares
        # 1 mV (vertical) = 10 mm and 0.2 sec (horizontal) = 5 mm in standard ECG
        # So the aspect ratio should be 0.5 (5mm/10mm) to make squares appear square
        # We're using time in seconds and voltage in mV, so this works out
        self.ax.set_aspect(0.2/0.5)  # This makes the grid squares perfectly square
        
        # Standard ECG grid specification:
        # 1mm squares (thin pink lines) = 0.04s x 0.1mV
        # 5mm squares (thick red lines) = 0.2s x 0.5mV
        
        # Configure grid visibility and style
        # First the minor grid (1mm x 1mm squares in pink) - 0.04s x 0.1mV
        self.ax.grid(True, which='minor', color='#FF9999', linestyle='-', linewidth=0.5, alpha=0.8)
        
        # Then the major grid (5mm x 5mm squares in red) - 0.2s x 0.5mV (draw on top of minor)
        self.ax.grid(True, which='major', color='#FF0000', linestyle='-', linewidth=1.0, alpha=0.9)
        
        # Set ticks for standard ECG paper
        # X-axis: 1 big square = 0.2s, 1 small square = 0.04s
        self.ax.xaxis.set_major_locator(MultipleLocator(0.2))  # 5mm squares horizontally
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.04))  # 1mm squares horizontally
        
        # Y-axis: 1 big square = 0.5mV, 1 small square = 0.1mV
        self.ax.yaxis.set_major_locator(MultipleLocator(0.5))   # 5mm squares vertically
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.1))   # 1mm squares vertically
        
        # Set standard ECG paper range (typically ±1.5 mV for normal display)
        # Typical clinical ECG shows approximately 3 large boxes (1.5mV) above and 1 large box (0.5mV) below baseline
        self.standard_y_min = -0.5  # 1 large square below baseline
        self.standard_y_max = 1.5   # 3 large squares above baseline
        
        # Force axis limits to ensure consistent display
        self.ax.set_ylim(self.standard_y_min, self.standard_y_max)
        
        # Style the plot to match clinical ECG format
        self.ax.set_title("Standard 12-Lead ECG Recording", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Time (sec)", fontsize=10)
        self.ax.set_ylabel("Voltage (mV)", fontsize=10)
        self.ax.tick_params(colors='black', labelsize=8)
        
        # Draw the baseline darker to make it stand out
        # Standard ECG baseline (0 mV line) is often highlighted
        self.baseline = self.ax.axhline(y=0, color='#444444', linestyle='-', linewidth=0.7, alpha=0.7)
        
        # Add time markers every second (vertical lines)
        # These are standard on many ECG recordings
        self.time_markers = []
        for i in range(10):
            marker = self.ax.axvline(x=i, color='#555555', linestyle='-', linewidth=0.5, alpha=0.5)
            self.time_markers.append(marker)
        
        # Add standard 1mV x 100ms calibration pulse at the beginning
        # This is the standardized calibration mark in clinical ECGs
        self.cal_rect = patches.Rectangle((0.05, -0.2), 0.1, 1.0, linewidth=1, 
                                         edgecolor='black', facecolor='none',
                                         transform=self.ax.transData)
        self.ax.add_patch(self.cal_rect)
        
        # Add standard ECG markers
        self.cal_text = self.ax.text(0.05, -0.3, "1 mV", fontsize=7, color='black', 
                                    fontweight='bold')
        
        # Add time scale indicators
        self.time_scale_text = self.ax.text(0.25, -0.4, "25 mm/sec", fontsize=7, color='black',
                                          fontweight='bold')
        
        # Add ECG standardization text - common on clinical ECGs
        self.standard_text = self.ax.text(0.02, 1.4, 
                                         "10 mm/mV, 25 mm/s", 
                                         fontsize=8, fontweight='bold', color='black')
        
        # Annotations for wave components
        self.labels = {
            'P': {'pos': None, 'text': None, 'color': '#0066CC'},
            'Q': {'pos': None, 'text': None, 'color': '#9900CC'},
            'R': {'pos': None, 'text': None, 'color': '#FF0000'},
            'S': {'pos': None, 'text': None, 'color': '#9900CC'},
            'T': {'pos': None, 'text': None, 'color': '#00CC00'}
        }
        
        # Standard ECG intervals text box
        self.interval_text = self.ax.text(0.02, 0.97, 
                              "P-R: --- ms\nQRS: --- ms\nQ-T: --- ms", 
                              transform=self.ax.transAxes, 
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'), 
                              fontsize=8, verticalalignment='top')
        
        # Add lead label
        self.lead_label = self.ax.text(0.98, 0.97, "Lead II", 
                          transform=self.ax.transAxes,
                          fontsize=10, fontweight='bold',
                          horizontalalignment='right',
                          verticalalignment='top')
        
        # Style the spines
        for spine in self.ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.5)
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # Add tight layout with padding
        self.fig.tight_layout(pad=2.0)
        
    def update_plot(self):
        """Update the plot with new data points using standard ECG format"""
        if not self.is_playing or self.current_index >= len(self.df):
            return
        
        # Add new data points
        for _ in range(min(self.data_increment, len(self.df) - self.current_index)):
            self.time_data.append(self.df.iloc[self.current_index, 0])
            self.ecg_data.append(self.df.iloc[self.current_index, 1])
            self.current_index += 1
        
        # If we have enough data, calculate heart rate and detect ECG components
        if len(self.ecg_data) > 100:
            self.calculate_heart_rate()
            self.detect_ecg_components()
        
        # Update the plot
        if len(self.time_data) > 0:
            # Determine the visible time window (standard ECG strips are typically 10 seconds)
            current_time = self.time_data[-1]
            start_time = max(0, current_time - self.window_size)
            
            # Get the visible data window
            visible_indices = [i for i, t in enumerate(self.time_data) if start_time <= t <= current_time]
            
            if visible_indices:
                visible_time = [self.time_data[i] for i in visible_indices]
                visible_ecg = [self.ecg_data[i] for i in visible_indices]
                
                # Update the line data
                self.line.set_data(visible_time, visible_ecg)
                
                # Adjust the plot x-axis limits
                self.ax.set_xlim(start_time, current_time)
                
                # Use fixed, standardized y-axis limits for ECG display
                # Standard is typically ±1.5mV from baseline
                self.ax.set_ylim(self.standard_y_min, self.standard_y_max)
                
                # Scale the display to standard 25mm/s paper speed (where 1mm = 0.04s)
                # and 10mm/mV amplitude (where 1mm = 0.1mV)
                # This is handled by the MultipleLocator settings in setup_plot
                
                # Update the wave segment annotations if available
                self.update_waveform_annotations(visible_time, visible_ecg)
                
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
            
    def detect_ecg_components(self):
        """Detect P, QRS, T waves and their intervals"""
        if len(self.ecg_data) < 200:  # Need enough data
            return
            
        # Get the most recent cardiac cycle
        data_window = 2  # seconds of data to analyze
        sampling_rate = 500  # estimated sampling rate
        
        # Use the last segment of data to find a complete cycle
        if len(self.ecg_data) > data_window * sampling_rate:
            recent_time = self.time_data[-int(data_window * sampling_rate):]
            recent_data = self.ecg_data[-int(data_window * sampling_rate):]
        else:
            recent_time = self.time_data
            recent_data = self.ecg_data
            
        # Find R peaks (main spike of QRS complex)
        if SCIPY_AVAILABLE:
            # Use scipy's find_peaks
            peaks, _ = find_peaks(recent_data, height=0.5, distance=sampling_rate*0.4)
            
            if len(peaks) >= 2:
                # We need at least 2 R peaks to define a complete cycle
                peak1_idx, peak2_idx = peaks[:2]
                r_peak1_time = recent_time[peak1_idx]
                r_peak1_val = recent_data[peak1_idx]
                
                # Mark the peaks
                self.labels['R']['pos'] = (r_peak1_time, r_peak1_val)
                
                # Calculate cycle length
                cycle_len = peaks[1] - peaks[0]
                
                # Look for P wave (typically 120-200ms before R)
                p_search_start = max(0, peak1_idx - int(0.2 * sampling_rate))
                p_search_end = max(0, peak1_idx - int(0.05 * sampling_rate))
                if p_search_start < p_search_end:
                    p_segment = recent_data[p_search_start:p_search_end]
                    if len(p_segment) > 0:
                        p_idx = p_search_start + np.argmax(p_segment)
                        self.labels['P']['pos'] = (recent_time[p_idx], recent_data[p_idx])
                
                # Look for Q wave (small negative deflection before R)
                q_search_start = max(0, peak1_idx - int(0.05 * sampling_rate))
                q_search_end = peak1_idx
                if q_search_start < q_search_end:
                    q_segment = recent_data[q_search_start:q_search_end]
                    if len(q_segment) > 0:
                        q_idx = q_search_start + np.argmin(q_segment)
                        self.labels['Q']['pos'] = (recent_time[q_idx], recent_data[q_idx])
                
                # Look for S wave (negative deflection after R)
                s_search_start = peak1_idx
                s_search_end = min(len(recent_data), peak1_idx + int(0.05 * sampling_rate))
                if s_search_start < s_search_end:
                    s_segment = recent_data[s_search_start:s_search_end]
                    if len(s_segment) > 0:
                        s_idx = s_search_start + np.argmin(s_segment)
                        self.labels['S']['pos'] = (recent_time[s_idx], recent_data[s_idx])
                
                # Look for T wave (typically 160-300ms after R peak)
                t_search_start = peak1_idx + int(0.16 * sampling_rate)
                t_search_end = min(len(recent_data), peak1_idx + int(0.3 * sampling_rate))
                if t_search_start < t_search_end and t_search_end < len(recent_data):
                    t_segment = recent_data[t_search_start:t_search_end]
                    if len(t_segment) > 0:
                        t_idx = t_search_start + np.argmax(t_segment)
                        self.labels['T']['pos'] = (recent_time[t_idx], recent_data[t_idx])
                    
                # Calculate intervals if we have the necessary components
                intervals_text = []
                
                # PR interval: beginning of P wave to beginning of QRS
                if self.labels['P']['pos'] and self.labels['Q']['pos']:
                    pr_interval = (self.labels['Q']['pos'][0] - self.labels['P']['pos'][0]) * 1000  # convert to ms
                    intervals_text.append(f"P-R: {pr_interval:.0f} ms")
                else:
                    intervals_text.append("P-R: --- ms")
                    
                # QRS duration
                if self.labels['Q']['pos'] and self.labels['S']['pos']:
                    qrs_duration = (self.labels['S']['pos'][0] - self.labels['Q']['pos'][0]) * 1000  # convert to ms
                    intervals_text.append(f"QRS: {qrs_duration:.0f} ms")
                else:
                    intervals_text.append("QRS: --- ms")
                    
                # QT interval
                if self.labels['Q']['pos'] and self.labels['T']['pos']:
                    qt_interval = (self.labels['T']['pos'][0] - self.labels['Q']['pos'][0]) * 1000  # convert to ms
                    intervals_text.append(f"Q-T: {qt_interval:.0f} ms")
                else:
                    intervals_text.append("Q-T: --- ms")
                    
                # Update interval text
                self.interval_text.set_text('\n'.join(intervals_text))
        
    def update_waveform_annotations(self, visible_time, visible_ecg):
        """Update the annotations for ECG waveform components if they're in view"""
        # Remove existing annotations
        for wave in self.labels.values():
            if wave['text'] is not None:
                wave['text'].remove()
                wave['text'] = None
        
        # Check which components are in the current view
        current_time_min = min(visible_time)
        current_time_max = max(visible_time)
        
        for wave_name, wave_data in self.labels.items():
            if wave_data['pos'] is not None:
                wave_time, wave_val = wave_data['pos']
                
                if current_time_min <= wave_time <= current_time_max:
                    # Add annotation for the wave component
                    wave_data['text'] = self.ax.annotate(
                        wave_name, 
                        (wave_time, wave_val),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        fontweight='bold',
                        color=wave_data['color'],
                        backgroundcolor='white',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none")
                    )

    def calculate_heart_rate(self):
        """Calculate and display heart rate from the ECG data using clinical standards"""
        try:
            # Use the last 10 seconds of data for calculation (standard approach for clinical ECG)
            window_size = 10  # seconds
            sampling_rate = 500  # Estimate from data density
            
            # Get the last section of data
            recent_data = self.ecg_data[-int(min(window_size * sampling_rate, len(self.ecg_data))):]
            recent_time = self.time_data[-int(min(window_size * sampling_rate, len(self.time_data))):]
            
            if len(recent_data) < 100:  # Need enough data
                return
            
            # Optional: Apply bandpass filter to clean the signal for better peak detection
            if SCIPY_AVAILABLE:
                try:
                    # Apply bandpass filter (5-20 Hz typically used for QRS detection)
                    nyquist = 0.5 * sampling_rate
                    low = 5 / nyquist
                    high = 20 / nyquist
                    b, a = butter(2, [low, high], btype='band')
                    filtered_data = filtfilt(b, a, recent_data)
                    
                    # Find R peaks (QRS complexes)
                    # Height is adaptive based on signal characteristics
                    # Distance set to avoid detecting T waves as peaks
                    height_threshold = np.mean(filtered_data) + 0.5 * np.std(filtered_data)
                    min_distance = int(sampling_rate * 0.4)  # Minimum 400ms between peaks (150 BPM max)
                    
                    peaks, _ = find_peaks(filtered_data, 
                                         height=height_threshold, 
                                         distance=min_distance)
                    
                    if len(peaks) > 1:
                        # Calculate RR intervals in seconds
                        rr_intervals = np.diff([recent_time[i] for i in peaks])
                        
                        # Filter out unrealistic intervals (clinical approach)
                        valid_intervals = [rr for rr in rr_intervals if 0.3 <= rr <= 2.0]  # 30-200 BPM range
                        
                        if len(valid_intervals) > 0:
                            # Calculate heart rate using the average RR interval
                            avg_rr = np.mean(valid_intervals)
                            self.heart_rate = int(60 / avg_rr)
                            
                            # Clinical quality check - ensure HR is in reasonable range
                            if 30 <= self.heart_rate <= 200:
                                # Update the display with calculated HR
                                self.hr_display.config(text=f"{self.heart_rate}")
                                
                                # Update status based on HR ranges
                                if self.heart_rate < 60:
                                    self.status_label.config(text="Status: Bradycardia", fg="#FF6600")
                                elif self.heart_rate > 100:
                                    self.status_label.config(text="Status: Tachycardia", fg="#FF0000")
                                else:
                                    self.status_label.config(text="Status: Normal Sinus Rhythm", fg="#009900")
                            else:
                                self.hr_display.config(text="--")
                                self.status_label.config(text="Status: Invalid HR Reading", fg="#FF0000")
                        else:
                            self.hr_display.config(text="--")
                            self.status_label.config(text="Status: Calculating...", fg="#666666")
                    else:
                        self.hr_display.config(text="--")
                        self.status_label.config(text="Status: Insufficient Data", fg="#666666")
                        
                except Exception as filter_error:
                    print(f"Error in filtering: {filter_error}")
                    # Fall back to simple calculation
                    self._simple_hr_calculation(recent_data, sampling_rate)
            else:
                # Simple fallback method if scipy is not available
                self._simple_hr_calculation(recent_data, sampling_rate)
                
        except Exception as e:
            print(f"Error calculating heart rate: {e}")
            self.hr_display.config(text="--")
            self.status_label.config(text="Status: Error", fg="#FF0000")
    
    def _simple_hr_calculation(self, recent_data, sampling_rate):
        """Basic heart rate calculation as fallback"""
        try:
            # Use basic threshold detection
            threshold = np.mean(recent_data) + 1.5 * np.std(recent_data)
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
                        # Add minimum distance check
                        if not peaks or (peak_idx - peaks[-1]) > sampling_rate * 0.4:
                            peaks.append(peak_idx)
            
            if len(peaks) > 1:
                # Calculate average interval between peaks
                avg_interval = np.mean(np.diff(peaks)) / sampling_rate  # in seconds
                self.heart_rate = int(60 / avg_interval)
                
                # Clinical quality check
                if 30 <= self.heart_rate <= 200:
                    # Update the display
                    self.hr_display.config(text=f"{self.heart_rate}")
                    
                    # Update status based on HR ranges
                    if self.heart_rate < 60:
                        self.status_label.config(text="Status: Bradycardia", fg="#FF6600")
                    elif self.heart_rate > 100:
                        self.status_label.config(text="Status: Tachycardia", fg="#FF0000")
                    else:
                        self.status_label.config(text="Status: Normal Rhythm", fg="#009900")
                else:
                    self.hr_display.config(text="--")
                    self.status_label.config(text="Status: Invalid Reading", fg="#FF0000")
            else:
                self.hr_display.config(text="--")
                self.status_label.config(text="Status: Calculating...", fg="#666666")
                
        except Exception as e:
            print(f"Simple HR calculation error: {e}")
            self.hr_display.config(text="--")
            self.status_label.config(text="Status: Error", fg="#FF0000")
    
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
        # Reset playback state
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        
        # Reset data
        self.current_index = 0
        self.time_data = []
        self.ecg_data = []
        self.filtered_ecg_data = []
        
        # Reset displays
        self.time_display.config(text=f"Time: 0.00s / {self.total_duration:.2f}s")
        self.hr_display.config(text="--")
        self.status_label.config(text="Status: Ready", fg="#003366")
        
        # Reset annotations
        if hasattr(self, 'labels'):
            for wave in self.labels.values():
                wave['pos'] = None
                if wave['text'] is not None:
                    wave['text'].remove()
                    wave['text'] = None
            
            # Reset interval text
            if hasattr(self, 'interval_text'):
                self.interval_text.set_text("P-R: --- ms\nQRS: --- ms\nQ-T: --- ms")
        
        # Reset the plot line
        if hasattr(self, 'line'):
            self.line.set_data([], [])
            
            # Make sure axes are set to standard clinical values
            if hasattr(self, 'ax'):
                # Reset to standard 10-second view
                self.ax.set_xlim(0, self.window_size)
                self.ax.set_ylim(self.standard_y_min, self.standard_y_max)
                
                # Make sure calibration is visible if enabled
                if hasattr(self, 'cal_rect'):
                    self.cal_rect.set_visible(self.calibration_pulse)
            
            self.canvas.draw()
    
    def update_speed_value(self, value):
        """Update playback speed based on slider value"""
        try:
            speed_factor = float(value)
            # Convert to a reasonable speed range (20ms to 200ms)
            self.update_speed = int(200 / speed_factor)
            self.data_increment = max(1, int(speed_factor))
            
            # Update the label if it exists
            if hasattr(self, 'speed_value_label'):
                self.speed_value_label.config(text=f"{speed_factor:.1f}x")
        except Exception as e:
            print(f"Error updating speed value: {e}")
    
    def update_window_size(self, value):
        """Update the time window size displayed"""
        try:
            self.window_size = float(value)
            
            # Update the label if it exists
            if hasattr(self, 'window_value_label'):
                self.window_value_label.config(text=f"{self.window_size:.1f} sec")
            
            # If we have data, update the plot
            if hasattr(self, 'line') and hasattr(self, 'ax') and len(self.time_data) > 0:
                # Update the plot with new window size
                current_time = self.time_data[-1]
                start_time = max(0, current_time - self.window_size)
                self.ax.set_xlim(start_time, current_time)
                self.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating window size: {e}")
            
    def on_resize(self, event):
        """Handle window resize events to maintain proper grid aspect ratio"""
        # Only process if the event is for the main window and we have an active plot
        if event.widget == self.master and hasattr(self, 'fig') and hasattr(self, 'ax'):
            # Give the UI a moment to settle with the new size
            self.master.after(200, self._update_plot_after_resize)

    def _update_plot_after_resize(self):
        """Update the plot after a window resize to maintain proper grid appearance"""
        if hasattr(self, 'fig') and hasattr(self, 'ax'):
            # Recalculate the aspect ratio based on the current speed and gain settings
            new_aspect = (0.2 * (25 / self.paper_speed)) / (0.5 * (10 / self.gain))
            self.ax.set_aspect(new_aspect)
            
            # Make sure axis limits are maintained
            if len(self.time_data) > 0:
                current_time = self.time_data[-1]
                start_time = max(0, current_time - self.window_size)
                self.ax.set_xlim(start_time, current_time)
            else:
                self.ax.set_xlim(0, self.window_size)
                
            self.ax.set_ylim(self.standard_y_min, self.standard_y_max)
            
            # Update the canvas
            self.fig.tight_layout(pad=2.0)
            self.canvas.draw()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()
