
import tkinter as tk

class VideoFeed(tk.Frame):
    def __init__(self, app, parent):
        super().__init__(parent)
        self.parent = app 
        self.setup_ui()

    def setup_ui(self):
        label = tk.Label(self, text="Video Feed Screen", font=("Arial", 24))
        label.pack(pady=50)

        # Add widgets for the logs screen
