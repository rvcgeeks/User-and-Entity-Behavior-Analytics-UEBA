
from os import path
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

Tk().withdraw()
filename = askopenfilename(initialdir = path.dirname(__file__), filetypes=[("Matplotlib Figures", ".fig.pickle")])
print(filename)

fig = pickle.load(open(filename, 'rb'))

dummy = plt.figure()
new_manager = dummy.canvas.manager
new_manager.canvas.figure = fig
fig.set_canvas(new_manager.canvas)

plt.show()
