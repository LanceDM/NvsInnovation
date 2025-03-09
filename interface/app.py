
from menu import Menu
from logs import Logs
from tkinter import messagebox
from videosetup import VideoSetup
from contactsetup import ContactSetup
from videofeed import VideoFeed

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Puranatics")
        self.root.geometry("500x500")

        # Initialize the frames 
        self.menu = Menu(self, self.root)
        self.logs = Logs(self, self.root)
        self.videosetup = VideoSetup(self, self.root)
        self.contactsetup = ContactSetup(self, self.root)
        self.videofeed = VideoFeed(self, self.root)
        self.current_panel = None  

    def switch_panel(self, panel_value):
        """Switch between frames based on panel_value"""
        if self.current_panel:
            self.current_panel.destroy()  
        # Switch frames based on the passed panel_value
        if panel_value == 1:
            self.current_panel = self.videofeed  # Show Video Feed panel
            self.current_panel.pack(fill="both", expand=True)
        elif panel_value == 2:
            self.current_panel = self.videosetup  # Show Video Setup panel
            self.current_panel.pack(fill="both", expand=True)
        elif panel_value == 3:
            self.current_panel = self.contactsetup  # Show Contact Setup panel
            self.current_panel.pack(fill="both", expand=True)
        elif panel_value == 4:
            self.current_panel = self.logs  # Show Logs panel
            self.current_panel.pack(fill="both", expand=True)
        elif panel_value == 0:
            self.current_panel = self.menu  # Show Menu panel
            self.current_panel.pack(fill="both", expand=True)
        
    def run(self):
        """Start with the Menu"""
        self.switch_panel(0)  

    def exit_system(self):
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
            self.root.quit()  