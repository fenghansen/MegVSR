import numpy as np
import cv2
import os
from data_process import get_frames_from_video

def rgb2gray(img):
    r = img[...,0]*0.299
    g = img[...,1]*0.587
    b = img[...,2]*0.114
    return r+g+b

def visualize(flow, name='flow'):
    h, w, c = flow.shape
    hsv = np.zeros((h,w,3), dtype=np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow(name, bgr)
    return bgr

def calOptflow(prvs_frame, next_frame, vis=False):
    hsv = np.zeros_like(prvs_frame)
    next_frame_gray = rgb2gray(next_frame)
    prvs_frame_gray = rgb2gray(prvs_frame)
    flow = cv2.calcOpticalFlowFarneback(prvs_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2,
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    if vis:
        bgr = visualize(flow, name='flow')
        I = np.mean(bgr,axis=-1)
        shape = I.shape
        total = 1
        for i, s in enumerate(shape):
            total *= s
        no_move = len(I[I==0])
        print(no_move, total, f"{no_move/total * 100:.2f}%")
        prvs_frame = prvs_frame[:,:,::-1]
        merge = prvs_frame//2 + bgr//2
        merge[I<1] = prvs_frame[I<1]
        cv2.imshow('ori', merge)
        k = cv2.waitKey(10) & 0xff
    return flow

if __name__ == '__main__':
    import h5py
    root_dir = r'F:\datasets\MegVSR\train_png\85.mkv_down4x.mp4_frames'
    files = [os.path.join(root_dir, name) for name in os.listdir(root_dir)]
    # imgs = get_frames_from_video(85)
    flows = []
    prvs_img = cv2.imread(files[0])[:,:,::-1]
    for i, file in enumerate(files[1:]):
        next_img = cv2.imread(file)[:,:,::-1]
        flow = calOptflow(prvs_img, next_img)
        flows.append(flow)
    flows.append(np.zeros_like(flow))
    flows = np.array(flows).astype(np.float32)
    # f = h5py.File(os.path.join(TRAIN_DATA_STORAGE,'MegVSR_train_%02d.h5' % frame_id), 'w')
    f = h5py.File('flows.h5', 'w')
    dst_hr = f.create_dataset('85', data=flows)
    f.close()
    # np.savez_compressed("flow_85.npz", flows)
    # cap = cv2.VideoCapture(cv2.samples.findFile(r"F:\datasets\MegVSR\train\85.mkv_down4x.mp4"))
    # # ret, frame1 = cap.read()
    # frame1 = imgs[0][:,:,::-1]
    # prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    # hsv = np.zeros_like(frame1)
    # hsv[...,1] = 255
    # for i, file in enumerate(files[:-1]):
    #     prvs_img = cv2.imread(files[i])
    #     next_img = cv2.imread(files[i+1])
    #     next_frame_gray = rgb2gray(next_img)
    #     prvs_frame_gray = rgb2gray(prvs_img)
        
    #     # ret, frame2 = cap.read()

    #     frame2 = imgs[i+1][:,:,::-1]
    #     next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #     print(np.array_equal(next, next_frame_gray))
    #     # imgs[i] = next_img
    #     # diff = cv2.normalize(np.abs(next - next_frame_gray),None,0,255,cv2.NORM_MINMAX) 
    #     # cv2.imshow('diff', diff)
    #     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 9, 3, 5, 1.2, 1)
    #     bgr = visualize(flow, name='flow3')
    #     I = np.mean(bgr,axis=-1)
    #     prvs_frame = frame2
    #     merge = prvs_frame//2 + bgr//2
    #     merge[I<1] = prvs_frame[I<1]
    #     cv2.imshow('ori', merge)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #     elif k == ord('s'):
    #         cv2.imwrite('opticalfb.png',frame2)
    #         cv2.imwrite('opticalhsv.png',bgr)
    #     prvs = next

