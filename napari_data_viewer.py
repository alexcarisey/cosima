import os
import json
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import ttk
import re
import napari
from PIL import Image
import numpy as np
import tifffile

class App(tk.Tk):

    def __init__(self):
        super().__init__()

        def exit_gui():
            file_name = folder_path.get().split("/")[-1]
            file_list = os.listdir(os.path.realpath(folder_path.get()))
            if not napari_running.get():
                napari_running.set(True)
                viewer = napari.Viewer()
                for f in file_list:
                    ext = os.path.splitext(f)[-1].lower()
                    if ext == '.tif':
                        # if f.find("id_layer"):
                        img = tifffile.imread(os.path.realpath(folder_path.get() + "/" + f))
                        # img = Image.open(os.path.realpath(folder_path.get() + "/" + f), mode="RGBA")
                        # img_np = np.array(img).astype(np.uint16, casting="same_kind")
                        viewer.add_image(img, name=f)
                viewer.layers
                napari.run()

        def get_input_path():
            INPUT_PATH = r'{}'.format(askdirectory(initialdir=r'./output', mustexist=True))
            folder_path.set(INPUT_PATH)
            print(INPUT_PATH)

        def get_current_value(part):
            return part.get()

        self.geometry('450x120')
        self.resizable(0, 0)
        self.title('COSIMA_img_viewer')

        # UI options
        paddings = {'padx': 5, 'pady': 5}
        entry_font = {'font': ('Helvetica', 11)}

        # configure the grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)

        napari_running = tk.BooleanVar(self, value=False)
        folder_path = tk.StringVar(self, value=r'./output')

        # heading
        heading = ttk.Label(self, text='Please select an image folder', style='Heading.TLabel')
        heading.grid(column=0, row=0, columnspan=3, pady=5, sticky=tk.N)

        # input path
        path_lb = ttk.Label(self, text="Input path: ")
        path_lb.grid(column=0, row=1, sticky=tk.W, **paddings)

        path_entry = ttk.Entry(self, textvariable=folder_path, state="disabled", **entry_font)
        path_entry.grid(column=1, row=1, sticky=tk.EW, **paddings)

        path_entry = ttk.Button(self, text="Select", command=get_input_path)
        path_entry.grid(column=2, row=1, sticky=tk.E, **paddings)

        # run button
        run_button = ttk.Button(self, text="Run", command=exit_gui)
        run_button.grid(column=2, row=99, sticky=tk.E, **paddings)

        # configure style
        self.style = ttk.Style(self)
        self.style.configure('TLabel', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11))

        # heading style
        self.style.configure('Heading.TLabel', font=('Helvetica', 12))


app = App()
app.mainloop()

