import os.path, time
import sys
import numpy as np
import cv2
import moviepy.editor as mpy

fname = ['IMG_0214.mov', 'IMG_1481.mov'][1]
vid = mpy.VideoFileClip(fname)
cnt = 0
x = 0



for i, (tstamp, frame) in enumerate(vid.iter_frames(with_times=True)):
    ts = (time.ctime(os.path.getmtime(fname)).split(':')[-1]).split(' ')[0]
    #print (ts)
    tstamp = tstamp%60
    sec = tstamp+int(ts)+cnt
    #print (sec)
    if sec >= 82:
        
        print(sec, i)
        
        if sec> 90 and x%50 == 0 and sec < 200:
            vid.save_frame("Output/img 0214-1481/Rframe"+str(x)+".png", t = (cnt/60,float(sec-int(ts)-cnt)))
        x +=1
    
    if tstamp > 59.99:
        cnt += 60
    if x > 1500:
        break

