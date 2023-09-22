import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils import *

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        load_image(file_path)

def upload_pt(mode:str):
    file_path = filedialog.askopenfilename(filetypes=[("pt", "*.pt")])
    load_pt(mode,file_path)

def load_image(file_path):
    global image_label, img
    img = Image.open(file_path)
    img = img.resize((300, 300))
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo

def recognize_text():
    global img
    if img is not None:
        img = np.array(img)
        try:
            res = main(img)
            text = f"{res}°"
        except Exception as e:
            text = f"识别失败,错误原因:{e}"
    else:
        text = "识别失败,图片存在问题"
    result_label.config(text=text)

root = tk.Tk()
root.title("图像识别")

upload_pt_button = tk.Button(root, text="上传仪表检测权重文件", command=lambda : upload_pt("detect"))
upload_pt_button.pack(pady=10)

upload_pt_button = tk.Button(root, text="上传倾斜校正权重文件", command=lambda : upload_pt("correct"))
upload_pt_button.pack(pady=10)

upload_pt_button = tk.Button(root, text="上传指针识别权重文件", command=lambda : upload_pt("read"))
upload_pt_button.pack(pady=10)

upload_image_button = tk.Button(root,text="上传待识别图片",command=upload_image)
upload_image_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

recognize_button = tk.Button(root, text="识别", command=recognize_text)
recognize_button.pack(pady=10)

result_label = tk.Label(root, wraplength=300, justify="left")
result_label.pack()

img = None

root.mainloop()
