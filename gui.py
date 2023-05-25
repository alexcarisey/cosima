import os
import sys
import json
import ast
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import ttk


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        def exit_gui():
            parameter_setting = {"INPUT_PATH": folder_path.get(),
                                 "ERODE_THICKNESS": erode_thickness.get(),
                                 "RING_THICKNESS": ring_thickness.get(),
                                 "CP_START": cp_start.get(),
                                 "CP_END": cp_end.get(),
                                 "BK_SUB": background_sub.get(),
                                 "SHOW_OVERLAP": show_overlap.get(),
                                 "SHOW_PLOTS": show_plots.get()
                                 }
            print(str(parameter_setting))

            if cp_end.get() > ring_thickness.get():
                err_msg_lb.configure(text="Wrong setting: compression end layer exceed thickness!")

            elif cp_start.get() > ring_thickness.get():
                err_msg_lb.configure(text="Wrong setting: compression starting layer exceed thickness!")

            elif cp_start.get() > cp_end.get():
                err_msg_lb.configure(text="Wrong setting: compression starting layer smaller than end layer!")

            else:
                err_msg_lb.configure(text="valid setting, proceed to analysis...")
                parameters = str(parameter_setting).replace(" ", "")
                os.system("python main.py " + json.dumps(parameters))
                self.destroy()

        def get_input_path():
            INPUT_PATH = r'{}'.format(askdirectory(initialdir=r'./input', mustexist=True))
            folder_path.set(INPUT_PATH)
            print(INPUT_PATH)

        def erode_slider_changed(event):
            erode_slide_num_lb.configure(text=get_current_value(erode_thickness))

        def ring_slider_changed(event):
            # ring_slider.get()
            # cp_start_max.set(ring_thickness.get())
            # print(cp_start_max.get())
            # print(int(event))
            ring_slide_num_lb.configure(text=get_current_value(ring_thickness))
            cp_start_slider.configure(to=ring_thickness.get())
            cp_start_slider.update_idletasks()
            cp_end.set(ring_thickness.get())
            cp_end_slider_num_lb.configure(text=get_current_value(cp_end))
            cp_end_slider.configure(to=ring_thickness.get())
            cp_end_slider.update_idletasks()


        def cp_start_slider_changed(event):
            # ring_slider.get()
            cp_end_slider.configure(from_=cp_start.get())
            cp_end_slider.update_idletasks()
            cp_start_slider_num_lb.configure(text=get_current_value(cp_start))

        def cp_end_slider_changed(event):
            # ring_slider.get()
            cp_end_slider_num_lb.configure(text=get_current_value(cp_end))
            cp_start_slider.configure(to=cp_end.get())
            cp_start_slider.update_idletasks()

        def get_current_value(part):
            return part.get()

        self.geometry('450x450')
        self.resizable(0, 0)
        self.title('COSIMA')

        # UI options
        paddings = {'padx': 5, 'pady': 5}
        entry_font = {'font': ('Helvetica', 11)}

        # configure the grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)

        folder_path = tk.StringVar(self, value=r'./input')
        erode_thickness = tk.IntVar(self, value=0)
        ring_thickness = tk.IntVar(self, value=1)
        cp_start = tk.IntVar(self, value=1)
        cp_start_max = tk.IntVar(self, value=20)
        cp_end = tk.IntVar(self, value=1)
        cp_end_max = tk.IntVar(self, value=1)
        cp_end_min = tk.IntVar(self, value=20)
        background_sub = tk.BooleanVar(self, value=True)
        show_overlap = tk.BooleanVar(self, value=True)
        show_plots = tk.BooleanVar(self, value=True)

        # heading
        heading = ttk.Label(self, text='Parameters Setup', style='Heading.TLabel')
        heading.grid(column=0, row=0, columnspan=3, pady=5, sticky=tk.N)

        # input path
        path_lb = ttk.Label(self, text="Input path: ")
        path_lb.grid(column=0, row=1, sticky=tk.W, **paddings)

        path_entry = ttk.Entry(self, textvariable=folder_path, state="disabled", **entry_font)
        path_entry.grid(column=1, row=1, sticky=tk.EW, **paddings)

        path_entry = ttk.Button(self, text="Select", command=get_input_path)
        path_entry.grid(column=2, row=1, sticky=tk.E, **paddings)

        # erode
        erode_lb = ttk.Label(self, text="Erode:")
        erode_lb.grid(column=0, row=2, sticky=tk.W, **paddings)

        erode_slider = ttk.Scale(
            self,
            from_=0,
            to=20,
            orient='horizontal',  # horizontal
            variable=erode_thickness,
            command=erode_slider_changed,
        )
        erode_slider.grid(column=1, row=2, sticky=tk.EW, **paddings)

        erode_slide_num_lb = ttk.Label(
            self,
            text=get_current_value(erode_thickness)
        )
        erode_slide_num_lb.grid(column=2, row=2, sticky=tk.W, **paddings)

        # thickness
        thickness_lb = ttk.Label(self, text="Thickness:")
        thickness_lb.grid(column=0, row=3, sticky=tk.W, **paddings)

        ring_slider = ttk.Scale(
            self,
            from_=1,
            to=20,
            orient='horizontal',  # horizontal
            variable=ring_thickness,
            command=ring_slider_changed,
        )
        ring_slider.grid(column=1, row=3, sticky=tk.EW, **paddings)

        ring_slide_num_lb = ttk.Label(
            self,
            text=get_current_value(ring_thickness)
        )
        ring_slide_num_lb.grid(column=2, row=3, sticky=tk.W, **paddings)

        # projection setting
        cp_start_lb = ttk.Label(self, text="Start of Compression:")
        cp_start_lb.grid(column=0, row=4, sticky=tk.W, **paddings)

        cp_start_slider = ttk.Scale(
            self,
            from_=1,
            to=cp_start_max.get(),
            orient='horizontal',  # horizontal
            variable=cp_start,
            command=cp_start_slider_changed,
        )
        cp_start_slider.grid(column=1, row=4, sticky=tk.EW, **paddings)

        cp_start_slider_num_lb = ttk.Label(
            self,
            text=get_current_value(cp_start)
        )
        cp_start_slider_num_lb.grid(column=2, row=4, sticky=tk.W, **paddings)

        cp_end_lb = ttk.Label(self, text="End of Compression:")
        cp_end_lb.grid(column=0, row=5, sticky=tk.W, **paddings)

        cp_end_slider = ttk.Scale(
            self,
            from_=1,
            to=20,
            orient='horizontal',  # horizontal
            variable=cp_end,
            command=cp_end_slider_changed,
        )
        cp_end_slider.grid(column=1, row=5, sticky=tk.EW, **paddings)

        cp_end_slider_num_lb = ttk.Label(
            self,
            text=get_current_value(cp_end)
        )
        cp_end_slider_num_lb.grid(column=2, row=5, sticky=tk.W, **paddings)

        # background sub
        bk_sub_lb = ttk.Label(self, text="Background subtraction:")
        bk_sub_lb.grid(column=0, row=6, columnspan=2, sticky=tk.W, **paddings)

        bk_sub_check = ttk.Checkbutton(
            self,
            variable=background_sub,

        )
        bk_sub_check.grid(column=1, row=6, sticky=tk.E, **paddings)

        # show overlap
        show_overlap_lb = ttk.Label(self, text="Show Overlapped pixel on plots:")
        show_overlap_lb.grid(column=0, row=7, columnspan=2, sticky=tk.W, **paddings)

        show_overlap_check = ttk.Checkbutton(
            self,
            variable=show_overlap,
        )
        show_overlap_check.grid(column=1, row=7, sticky=tk.E, **paddings)

        # show overlap
        show_plots_lb = ttk.Label(self, text="Display plots:")
        show_plots_lb.grid(column=0, row=79, columnspan=2, sticky=tk.W, **paddings)

        show_plots_check = ttk.Checkbutton(
            self,
            variable=show_plots,
        )
        show_plots_check.grid(column=1, row=79, sticky=tk.E, **paddings)

        # error message
        err_msg_lb = ttk.Label(self, text="")
        err_msg_lb.grid(column=0, row=89, columnspan=3, sticky=tk.W, **paddings)

        # run button
        login_button = ttk.Button(self, text="Run", command=exit_gui)
        login_button.grid(column=2, row=99, sticky=tk.E, **paddings)

        # configure style
        self.style = ttk.Style(self)
        self.style.configure('TLabel', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11))

        # heading style
        self.style.configure('Heading.TLabel', font=('Helvetica', 12))


app = App()
app.mainloop()

