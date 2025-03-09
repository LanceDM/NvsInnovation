
import tkinter as tk

class VideoSetup(tk.Frame):
    def __init__(self, app, parent):
        super().__init__(parent)
        self.parent = app 
        self.setup_ui()

    def setup_ui(self):
        label = tk.Label(self, text="Video Setup Screen", font=("Arial", 24))
        label.pack(pady=50)

