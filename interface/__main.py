# executable.py
import tkinter as tk
from app import App

def main():
    root = tk.Tk()  
    app = App(root)  
    app.run()  
    root.mainloop()  

if __name__ == "__main__":
    main()
