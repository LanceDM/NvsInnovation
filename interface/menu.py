# menu.py
import tkinter as tk
from tkinter import messagebox

class Menu(tk.Frame):
    def __init__(self, app, parent):
        super().__init__(parent)
        self.parent = app  # Set self.parent to the App instance
        self.setup_ui()

    def setup_ui(self):
        """Set up UI components like buttons and labels"""
        label = tk.Label(self, text="Main Menu", font=("Arial", 24))
        label.pack(pady=50)

        # Buttons to navigate to different panels
        videofeed_button = tk.Button(self, text="Go to Video Feed", font=("Arial", 16), command=self.go_to_videofeed)
        videofeed_button.pack(pady=10)

        videosetup_button = tk.Button(self, text="Go to Video Setup", font=("Arial", 16), command=self.go_to_videosetup)
        videosetup_button.pack(pady=10)

        contactsetup_button = tk.Button(self, text="Go to Contact Setup", font=("Arial", 16), command=self.go_to_contactsetup)
        contactsetup_button.pack(pady=10)

        logs_button = tk.Button(self, text="Go to Logs", font=("Arial", 16), command=self.go_to_logs)
        logs_button.pack(pady=10)

        exit_button = tk.Button(self, text="Exit", font=("Arial", 16), command=self.exit_system)
        exit_button.pack(pady=10)

    def go_to_videofeed(self):
        """Switch to Video Feed panel (pass value 1 to switch to Video Feed)"""
        self.parent.switch_panel(1)

    def go_to_videosetup(self):
        """Switch to Video Setup panel (pass value 2 to switch to Video Setup)"""
        self.parent.switch_panel(2)

    def go_to_contactsetup(self):
        """Switch to Contact Setup panel (pass value 3 to switch to Contact Setup)"""
        self.parent.switch_panel(3)

    def go_to_logs(self):
        """Switch to Logs panel (pass value 4 to switch to Logs)"""
        self.parent.switch_panel(4)

    def exit_system(self):
        self.parent.exit_system()

        

