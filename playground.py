import os
import sys
import copy
from time import time
import re
import math
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import mpl_interactions.ipyplot as iplt
from PIL import Image
import shutil
from scipy import interpolate
import os
import time
import threading
# importing the required modules
from tkinter import *  # importing all the widgets and modules from tkinter
import os  # importing the os module
import shutil  # importing the shutil module
from tkinter.ttk import Progressbar
from tkinter.filedialog import askdirectory
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
# folder = r'{}'.format(askdirectory())
# print(folder)
# print(os.path.realpath(folder))


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        def exit_gui():
            print(folder_path.get(), ring_thickness.get())
            print("parameter collected, extracting data...")
            # os.system("python main.py")
            # sys.exit()

        def get_input_path():
            INPUT_PATH = r'{}'.format(askdirectory())
            folder_path.set(INPUT_PATH)
            print(INPUT_PATH)

        def ring_slider_changed(event):
            # ring_slider.get()
            value_label.configure(text=get_current_value())

        def get_current_value():
            return ring_thickness.get()

        self.geometry('450x210')
        self.resizable(0, 0)
        self.title('COSIMA')

        # UI options
        paddings = {'padx': 5, 'pady': 5}
        entry_font = {'font': ('Helvetica', 11)}

        # configure the grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)

        folder_path = tk.StringVar()
        ring_thickness = tk.IntVar()

        # heading
        heading = ttk.Label(self, text='Parameters Setup', style='Heading.TLabel')
        heading.grid(column=0, row=0, columnspan=3, pady=5, sticky=tk.N)

        # username
        path_lb = ttk.Label(self, text="Input path: ")
        path_lb.grid(column=0, row=1, sticky=tk.W, **paddings)

        path_entry = ttk.Entry(self, textvariable=folder_path, state="disabled", **entry_font)
        path_entry.grid(column=1, row=1, sticky=tk.EW, **paddings)

        path_entry = ttk.Button(self, text="Select", command=get_input_path)
        path_entry.grid(column=2, row=1, sticky=tk.E, **paddings)

        # password
        thickness_lb = ttk.Label(self, text="Thickness:")
        thickness_lb.grid(column=0, row=2, sticky=tk.W, **paddings)

        ring_slider = ttk.Scale(
            self,
            from_=0,
            to=20,
            orient='horizontal',  # horizontal
            variable=ring_thickness,
            command=ring_slider_changed,
        )
        ring_slider.grid(column=1, row=2, sticky=tk.EW, **paddings)

        value_label = ttk.Label(
            self,
            text=get_current_value()
        )
        value_label.grid(column=2, row=2, sticky=tk.W, **paddings)
        # login button
        login_button = ttk.Button(self, text="Run", command=exit_gui)
        login_button.grid(column=2, row=3, sticky=tk.E, **paddings)

        # configure style
        self.style = ttk.Style(self)
        self.style.configure('TLabel', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11))

        # heading style
        self.style.configure('Heading.TLabel', font=('Helvetica', 12))


app = App()
app.mainloop()
