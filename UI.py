import tkinter as tk
import center_tk_window
from tkinter import filedialog
from Drowsiness_Detection import *

window = tk.Tk()
window.title('Driver Drowsiness Detection')

lbTitle = tk.Label(window, text="DRIVER DROWSINESS DETECTION",
                   font=("Arial Bold", 17))
lbTitle.place(x=5, y=1)

lbAuthor = tk.Label(window, text="Author: Ho Minh Canh",
                    font=("Arial Italic", 10))
lbAuthor.configure(underline=True)
lbAuthor.place(x=127, y=320)

EYE_AR_THRESH = tk.Label(window, text="Threshold:").place(x=36, y=50)
EYE_AR_THRESH_AREA = tk.Entry(window, width=11)
EYE_AR_THRESH_AREA.insert(0, "0.25")
EYE_AR_THRESH_AREA.place(x=96, y=50)

EYE_AR_CONSEC_FRAMES = tk.Label(window, text="Frames:").place(x=235, y=50)
EYE_AR_CONSEC_FRAMES_AREA = tk.Entry(window, width=12)
EYE_AR_CONSEC_FRAMES_AREA.insert(0, "40")
EYE_AR_CONSEC_FRAMES_AREA.place(x=285, y=50)


def onStreamAction(event=None):
    EYE_AR_THRESH_AREA_VALUES = EYE_AR_THRESH_AREA.get()
    EYE_AR_CONSEC_FRAMES_AREA_VALUES = EYE_AR_CONSEC_FRAMES_AREA.get()

    onStream(float(EYE_AR_THRESH_AREA_VALUES),
             int(EYE_AR_CONSEC_FRAMES_AREA_VALUES))


def openVideoAction(event=None):
    filename = filedialog.askopenfilename()
    EYE_AR_THRESH_AREA_VALUES = EYE_AR_THRESH_AREA.get()
    EYE_AR_CONSEC_FRAMES_AREA_VALUES = EYE_AR_CONSEC_FRAMES_AREA.get()
    onVideo(filename, float(EYE_AR_THRESH_AREA_VALUES),
            int(EYE_AR_CONSEC_FRAMES_AREA_VALUES))


streamBtn = tk.Button(window, text='Open Webcam ',
                      command=onStreamAction, bg="#ffff00", fg='black', font="Bold").place(x=42, y=90)

videoBtn = tk.Button(window, text='Open Video File',
                     command=openVideoAction, bg="#23D215", fg='black', font="Bold").place(x=235, y=90)

lbDetailThreshold = tk.Label(
    window, text="(*)Threshold: If the eye aspect ratio falls below this threshold, the system \n start counting the number of frames the person has closed their eyes for. \n (Default = 0.25)", fg='#0000ff').place(x=10, y=150)
lbDetailFrames = tk.Label(window, text="(*)Frames: If the number of frames the person has closed their eyes in \n exceeds '"'Frames'"', the system will sound an alarm to alerts the driver. \n (30 frames = 1 second, default = 40 frames)", fg='#0000ff').place(x=10, y=210)


closing_button = tk.Button(
    window, text="Close", command=window.destroy, bg="red").place(x=180, y=280)


window_width = 410
window_height = 320
window.geometry(str(window_width)+"x"+str(window_height)+'+470+180')
window.mainloop()
