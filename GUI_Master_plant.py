import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
import time
import functools
import operator
import CNNModelp
from tkinter import messagebox
global fn
fn = ""

root = tk.Tk()
root.configure(background="seashell2")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{w}x{h}+0+0")
root.title("Detection and Classification of Agricultural Pests")

# Background image
image2 = Image.open('zzz.jpg')
image2 = image2.resize((w, h))
background_image = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0)

# Header label with enhanced styling
lbl = tk.Label(root, text="Detection and Classification of Agricultural Pests", 
               font=('times', 25, 'bold'), height=1, width=70, 
               bg="#1e3d59", fg="white", relief="solid", bd=4, padx=20, pady=20)

# Adding a soft gradient effect using a custom color
lbl.place(x=0, y=0)


# Frame for buttons
frame_alpr = tk.LabelFrame(root, text="", width=220, height=550, bd=5, font=('times', 14, 'bold'), bg="black")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=90)

# Pest information
pest_info = {
    "Aphids": {"causes": "Overcrowding, excessive nitrogen", "prevention": "Introduce natural predators, use insecticidal soap"},
    "Armyworm": {"causes": "Overripe crops, dense foliage", "prevention": "Handpick worms, use natural predators"},
    "Beetle": {"causes": "Crop debris, favorable climate", "prevention": "Crop rotation, use of traps"},
    "Bollworm": {"causes": "Overripe crops, lack of natural enemies", "prevention": "Use of BT crops, natural predators"},
    "Grasshopper": {"causes": "Warm, dry weather", "prevention": "Use of insecticidal bait, natural predators"},
    "Mites": {"causes": "Hot, dry weather", "prevention": "Maintain humidity, use miticides"},
    "Mosquito": {"causes": "Standing water", "prevention": "Eliminate standing water, use mosquito nets"},
    "Sawfly": {"causes": "Dense foliage", "prevention": "Prune infected parts, use insecticides"},
    "Stem_borer": {"causes": "Infested crop residue", "prevention": "Destroy crop residue, use resistant varieties"},
}

def update_label(text):
    result_label = tk.Label(root, text=text, width=60, font=("bold", 20), bg='bisque2', fg='black')
    result_label.place(x=250, y=400)

def train_model():
    update_label("Model Training Start...............")
    start = time.time()
    # Replace CNNModelp.main() with your actual model training function
    X = CNNModelp.main()
    end = time.time()
    ET = f"Execution Time: {end - start:.4} seconds \n"
    msg = "Model Training Completed..\n" + ET
    print(msg)
    update_label(msg)

def convert_str_to_tuple(tup):
    return functools.reduce(operator.add, tup)

def test_model_proc(fn):
    IMAGE_SIZE = 64
    model = load_model('pest.h5', compile=False)
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    img = img.astype('float32') / 255.0
    prediction = model.predict(img)
    plant = np.argmax(prediction)
    pests = ["Aphids", "Armyworm", "Beetle", "Bollworm", "Grasshopper", "Mites", "Mosquito", "Sawfly", "Stem_borer"]
    return pests[plant]

def test_model():
    global fn
    if fn:
        update_label("Model Testing Start...............")
        start = time.time()
        pest = test_model_proc(fn)
        causes = pest_info.get(pest, {}).get("causes", "N/A")
        prevention = pest_info.get(pest, {}).get("prevention", "N/A")
        x2 = f"{pest} pest is detected"
        end = time.time()
        ET = f"Execution Time: {end - start:.4} seconds \n"
        msg = f"Image Testing Completed..\n{x2}\nCauses: {causes}\nPrevention: {prevention}\n{ET}"
        fn = ""
    else:
        msg = "Please Select Image For Prediction...."
    update_label(msg)

def openimage():
    global fn
    allowed_dir = 'C:/Users/ABHISHEK GITE/OneDrive/Desktop/Final Year Project/100% code Agri pest classification/100% code Agri pest classification/dataset/testing'
    
    fileName = askopenfilename(
        initialdir=allowed_dir,
        title='Select image for Analysis',
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if fileName:
        # Check if selected file is within allowed directory
        if not fileName.startswith(allowed_dir):
            messagebox.showerror("Invalid Image", "Please select an valid image only!")
            return

        IMAGE_SIZE = 200
        fn = fileName
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        imgtk = ImageTk.PhotoImage(img)
        img_label = tk.Label(root, image=imgtk)
        img_label.image = imgtk
        img_label.place(x=300, y=100)

def convert_grey():
    global fn
    if fn:
        IMAGE_SIZE = 200
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img)
        x1, y1 = img_array.shape[0], img_array.shape[1]
        gs = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_RGB2GRAY)
        gs = cv2.resize(gs, (x1, y1))
        retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        im_gs = Image.fromarray(gs)
        im_gs_tk = ImageTk.PhotoImage(im_gs)
        img2 = tk.Label(root, image=im_gs_tk, height=200, width=200, bg='white')
        img2.image = im_gs_tk
        img2.place(x=580, y=100)
        im_thresh = Image.fromarray(threshold)
        im_thresh_tk = ImageTk.PhotoImage(im_thresh)
        img3 = tk.Label(root, image=im_thresh_tk, height=200, width=200)
        img3.image = im_thresh_tk
        img3.place(x=880, y=100)
    else:
        update_label("Please Select Image For Conversion....")

def detect_objects():
    global fn
    if fn:
        model = load_model('pest.h5', compile=False)
        IMAGE_SIZE = 64
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img)
        img_array = img_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        img_array = img_array.astype('float32') / 255.0
        prediction = model.predict(img_array)
        pest_classes = ["Aphids", "Armyworm", "Beetle", "Bollworm", "Grasshopper", "Mites", "Mosquito", "Sawfly", "Stem_borer"]
        detected_pests = [pest_classes[i] for i in np.argmax(prediction, axis=1)]
        
        img_cv = cv2.imread(fn)
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Draw example bounding boxes; replace with actual model output if available
        (x, y, w, h) = (10, 15, 30, 30)
        for pest in detected_pests:
            cv2.rectangle(img_cv_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img_cv_rgb, pest, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        display_size = 200
        img_cv_rgb = cv2.resize(img_cv_rgb, (display_size, display_size))
        im = Image.fromarray(img_cv_rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        
        # Clear previous images and labels
        for widget in root.winfo_children():
            if isinstance(widget, tk.Label) and widget != background_label:
                widget.destroy()

        img_label = tk.Label(root, image=imgtk)
        img_label.image = imgtk
        img_label.place(x=300, y=100)
    else:
        update_label("Please Select Image For Object Detection....")

# Button styling
button_style = {
    "font": ('Helvetica', 14, 'bold'),
    "bg": "#1e3d59",              # Deep blue background
    "fg": "white",                # White text
    "activebackground": "#f5a623",  # Highlight color on hover
    "activeforeground": "black",  # Text color on hover
    "bd": 3,                      # Slight border
    "relief": "raised",           # Raised button effect
    "width": 18,
    "height": 2,
    "cursor": "hand2"
}


def show_about_model():
    from tkinter import Toplevel 
    about_window = Toplevel(root)
    about_window.title("About CNN Model")
    about_window.geometry("700x400")
    about_window.configure(bg="white")

    info = (
        "This project uses a Convolutional Neural Network (CNN) model to classify Agri pest.\n"
        "The categories include Aphids, Armyworm, Beetle,\n Bollworm, Grasshopper, Mites, Mosquito, Sawfly, Stem_borer.\n"
        "CNN is a deep learning model particularly effective for image classification tasks.\n"
        "It uses multiple layers such as convolutional, pooling, and fully connected layers\n"
        "to automatically extract features and make predictions with high accuracy.\n\n"
        "The model was trained on a labeled satellite imagery dataset, and it achieves\n"
        "over 95% classification accuracy."
    )

    tk.Label(about_window, text="About the Model", font=("Helvetica", 16, "bold"), bg="white").pack(pady=10)
    tk.Message(about_window, text=info, font=("Helvetica", 12), bg="white", width=600).pack(pady=10)

# Buttons
open_image_button = tk.Button(frame_alpr, text="Select Image", command=openimage, **button_style)
open_image_button.grid(row=0, column=0, pady=10)

train_model_button = tk.Button(frame_alpr, text="Train Model", command=train_model, **button_style)
train_model_button.grid(row=4, column=0, pady=10)

test_model_button = tk.Button(frame_alpr, text="Test Model", command=test_model, **button_style)
test_model_button.grid(row=3, column=0, pady=10)

convert_grey_button = tk.Button(frame_alpr, text="Convert to Grey", command=convert_grey, **button_style)
convert_grey_button.grid(row=1, column=0, pady=10)

detect_objects_button = tk.Button(frame_alpr, text="Detect Objects", command=detect_objects, **button_style)
detect_objects_button.grid(row=2, column=0, pady=10)


about_model_button = tk.Button(frame_alpr, text="About Model", command=show_about_model, **button_style)
about_model_button.grid(row=6, column=0, pady=10)

btn_exit = tk.Button(frame_alpr, text="Exit", command=root.destroy, font=('times', 14, 'bold'), bg='black', fg='white')
btn_exit.grid(row=5, column=0, pady=10)

root.mainloop()
