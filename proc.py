from threading import Thread
import playsound

def sound_alarm(path):
    playsound.playsound(path)

def playSound(path):
    t = Thread(target=sound_alarm, args=(path,), daemon=True)
    t.start()
