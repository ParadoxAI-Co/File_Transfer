from tkinter import ttk
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from clint.textui import progress
import random
import requests
import os
root = tk.Tk()
root.geometry("200x80")
root.title("File Transfer")
name_var=tk.StringVar()
def recive():
    root.withdraw()
    def r():
        name=name_entry.get()
        url = "https://paradoxai-co.github.io/File_Transfer/{}".format(name)
        r = requests.get(url, stream=True)
        path = name
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    f.write(chunk)
                    f.flush()
    root2=tk.Tk()
    root2.geometry("200x80")
    root.title("recive file")
    name_entry = tk.Entry(root2,textvariable = name_var, font=('calibre',10,'normal'))
    def disable_event():
       root.deiconify()
       root2.destroy()
    root2.protocol("WM_DELETE_WINDOW", disable_event)
    r = ttk.Button(root2, text = 'recive', command = r)
    name_entry.pack()
    r.pack()
    root2.mainloop()
    name_var.set("")
def send():
    os.system("start https://github.com/ParadoxAI-Co/File_Transfer/upload/main")

oper = tk.Label(root, text = 'choose your operation', font = ('calibre',10,'bold'))
recive = ttk.Button(root, text = 'recive file', command = recive)
send = ttk.Button(root, text = 'send file', command = send)
oper.pack()
recive.pack()
send.pack()
root.mainloop()