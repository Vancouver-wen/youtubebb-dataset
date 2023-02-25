import csv
from pytube import YouTube
import pytube.exceptions
import os
from tqdm import tqdm
import cv2 
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED, as_completed
import random
import time

def get_csv(filename): #return pairs
    keys_set=[]
    with open(filename)as f:
        f_csv = csv.reader(f)
        keys=set()
        times=[]
        previous_key=''
        for i,row in enumerate(f_csv):
            key=row[0]
            time=row[1]
            if key in keys:
                times.append(time)
            else:
                if i != 0:
                    keys_set.append((previous_key,times))
                    times=[]
                    keys.add(key)
                    times.append(time)
                    #import pdb;pdb.set_trace()
                else: # i ==0  起步阶段
                    keys.add(key)
                    times.append(time)
            previous_key=key
    return keys_set
def single_download(key,times,output_path,max_retries): # return rec,key,times
    filename=key+".mp4"
    rec=False
    for i in range(4):
        if i%4==0:
            #print(key," download from http invidious")
            url= 'http://invidious.epicsite.xyz/watch?v='+key 
        elif i%4==1:
            #print(key," download from https invidious")
            url='https://invidious.epicsite.xyz/watch?v='+key
        elif i%4==2:
            #print(key," download from http youtube")
            url='http://www.youtube.com/watch?v='+key
        elif i%4==3:
            #print(key," download from https youtube")
            url='https://www.youtube.com/watch?v='+key
        #视频下载成功就退出循环
        if os.path.exists(output_path+'/'+filename):
            #print(filename,' exists') if i==0 else print(filename,' have downloaded')
            rec=True
            return rec,key,times
        try:
            yt = YouTube(url)
            #print(f'Downloading video: {url}')
            #yt.streams.first().download()
            yt_out=yt.streams.filter(file_extension='mp4').get_highest_resolution()
            #print("=> yt_out: ",list(yt_out))
            yt_out.download(output_path=output_path,filename=filename,max_retries=max_retries)
        except pytube.exceptions.VideoUnavailable:
            #print(f'Video {url} is unavaialable, skipping.')
            return rec,key,times
        except Exception:
            #print(f'Video {url} fail, try again')
            time.sleep(random.randint(1, 6))
    return rec,key,times
def single_handle(pair,output_path,remove_raw_video):
    key,times=pair
    filename=output_path+'/'+key+".mp4"
    if os.path.exists(filename):
        try:
            cameraCapture = cv2.VideoCapture(filename)
            if not cameraCapture.isOpened():
                #处理mov,mp4,m4a,3gp,3g2,mj2 @ 00000160eb7e2040] moov atom not found
                return filename
            #创建图片的存储路径
            image_path=output_path+'/'+key+'/'
            print(image_path,' exists') if os.path.exists(image_path) else os.mkdir(image_path)
            #根据最近邻时间提取图片
            time_index=0
            #import pdb;pdb.set_trace()
            while time_index<len(times):
                success, frame = cameraCapture.read() #处理 \
                #[mov,mp4,m4a,3gp,3g2,mj2 @ 000001c906f65a80] stream 1, offset 0x1139683e: partial file
                #因为这通常表示文件不全，直接跳过该帧就可以了
                if success is False:
                    return filename
                milliseconds = int(cameraCapture.get(cv2.CAP_PROP_POS_MSEC))
                # if success is False:
                #     import pdb;pdb.set_trace()
                #print(milliseconds,"---",int(times[time_index]))
                if milliseconds>int(times[time_index]):
                    #print(milliseconds,"---",int(times[time_index]))
                    img_name=image_path+times[time_index]+'.jpg'
                    cv2.imwrite(img_name,frame)
                    time_index+=1
        except Exception:
            remove_raw_video=False
        else: #当 try 块没有出现异常时，程序会执行 else 块
            cameraCapture.release()
        #是否删除原视频
        if remove_raw_video:
            os.remove(filename)
        return None
    else:
        return filename
def single_process(pair,output_path,max_retries,remove_raw_video):
    rec,key,_=single_download(pair[0],pair[1],output_path,max_retries)
    if rec is True:
        video_borken=single_handle(pair,output_path,remove_raw_video)
        if video_borken is None:
            return None  
        else:
            print(f'Video {key} downloaded has broken')
            rec=False
            return (rec,video_borken)
    else:
        print(f'Video {key} fail to download')
        return (rec,key) #未处理成功，因为无法下载视频

def thread_process(pairs,output_path,max_retries,max_workers,remove_raw_video):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        all_task = []
        for pair in pairs:
            all_task.append(pool.submit(single_process,pair,output_path,max_retries,remove_raw_video))
        debug_data=[]
        with tqdm(total=100) as pbar:
            proportion=100/len(pairs)
            for out in as_completed(all_task):
                single_result=out.result()
                if single_result is not None:
                    debug_data.append(single_result) 
                pbar.update(proportion)
        wait(all_task, return_when=ALL_COMPLETED)
    #print("----complete-----")
    return debug_data
def all_process(csv_name,output_path,max_workers,max_retries,remove_raw_video,test):
    #import pdb;pdb.set_trace()
    pairs=get_csv(csv_name)
    print("=> read csv file successfully ..")
    if test:
        pairs=pairs[0:10]
    print("=> multi-thread downloading and processing ..")
    debug_data=thread_process(pairs,output_path,max_retries,max_workers,remove_raw_video)
    return debug_data
def save_debug(filepath,debug):
    filename = open(filepath, 'w')
    for value in debug:
        filename.write(str(value))
        filename.close()

if __name__=='__main__':
    csv_name='youtube_boundingboxes_detection_validation.csv'
    output_path='./video'
    max_workers=2  # 线程池最大容量
    max_retries=2  # url下载最大尝试次数
    test=False  # test只会下载前10个视频用于代码debug
    remove_raw_video=True  #将视频处理成图片后，删除原视频
    debug_path="debug.txt"
    debug_data=all_process(csv_name,output_path,max_workers,max_retries,remove_raw_video,test)
    #print(debug_path,' exists') if os.path.exists(debug_path) else os.mknod(debug_path)
    if os.path.exists(debug_path) is False:
        os.mknod(debug_path)
    save_debug(debug_path,debug_data)