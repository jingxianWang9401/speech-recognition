# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:02:41 2019

@author: wangjingxian
"""

import wave
import matplotlib.pyplot as plt
import numpy as np
import os


path = "E://speech_rocognition_demo//method2//dataset//noise//dakaicaidan" #添加路径

files= os.listdir(path) #得到文件夹下的所有文件名称  

files = [path + "//" + f for f in files if f.endswith('.wav')]


def wavtoimage():
    for i in range(len(files)):
        FileName = files[i]
        print("Transfer File Name is ",FileName)
        image_name=FileName.split('/')[-1].split('.')[0]
        f = wave.open(r"" + FileName, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)#读取音频，字符串格式
        waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
        waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
        waveData = np.reshape(waveData,[nframes,nchannels]).T
        f.close()
        # plot the wave
        plt.specgram(waveData[0],Fs = framerate, scale_by_freq = True, sides = 'default')
        plt.ylabel('Frequency(Hz)')
        plt.xlabel('Time(s)')
        plt.savefig('E://speech_rocognition_demo//method2//dataset//noise_images//dakaicaidan'+
                '//'+str(image_name)+'.jpg')
        plt.figure()
        

if __name__ == '__main__' :
    wavtoimage()
    print("Run Over")
     