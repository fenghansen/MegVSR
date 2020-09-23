import numpy as np
import cv2
import os
#from data_process import get_frames_from_video

def rgb2gray(img):
    r = img[...,0]*0.299
    g = img[...,1]*0.587
    b = img[...,2]*0.114
    return r+g+b

def visualize(flow, name='flow', show=True):
    h, w, c = flow.shape
    hsv = np.zeros((h,w,3), dtype=np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    if show:
        cv2.imshow(name, bgr)
    return bgr, np.mean(mag)*100

def calOptflow(prvs_frame, next_frame, vis=False, window_size=15):
    hsv = np.zeros_like(prvs_frame, dtype=np.float32)
    next_frame_gray = rgb2gray(next_frame).astype(np.float32)
    prvs_frame_gray = rgb2gray(prvs_frame).astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(prvs_frame_gray, next_frame_gray, None, 0.5, 3, window_size, 3, 5, 1.2, 1)
    if vis:
        bgr, ratio1 = visualize(flow, name='flow')
        I = np.mean(bgr,axis=-1)
        shape = I.shape
        total = 1
        for i, s in enumerate(shape):
            total *= s
        no_move = len(I[I==0])
        
        prvs_frame = prvs_frame[:,:,::-1]
        next_frame_fake = FlowShift(prvs_frame, flow)
        _, ratio2 = visualize(calOptflow(next_frame, next_frame_fake), name='differ')

        print(no_move, total, f"{no_move/total * 100:.2f}%, ratio: [{ratio1:.2f}] | [{ratio2:.2f}]")
        merge = next_frame_fake*.5 + bgr*.5
        merge[I<1] = next_frame_fake[I<1]
        cv2.imshow('new_frames', np.uint8(merge))
        k = cv2.waitKey(30) & 0xff
    return flow

def FlowSplit(flow):
    fm = [[None]*2, [None]*2]
    wm = [[None]*2, [None]*2]
    weights = [[None]*2, [None]*2]
    flows = [[None]*2, [None]*2]
    for i in range(2):
        # xf, xc
        # yf, yc
        fm[i][0] = np.floor(flow[:,:,i])
        fm[i][1] = np.ceil(flow[:,:,i])

    for i in range(2):
        for j in range(2):
            wm[i][j] = np.abs(flow[:,:,i] - fm[i][1-j])
    for i in range(2):
        for j in range(2):
            weights[i][j] = wm[0][i] * wm[1][j]
    for i in range(2):
        for j in range(2):
            flows[i][j] = np.stack((fm[0][i], fm[1][j]), axis=-1)
    return flows, np.expand_dims(np.array(weights),axis=-1)

def FlowShift(image, flow, weight=1):
    h, w, c = flow.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    pixel_map = coords - flow
    new_img = cv2.remap(image, pixel_map, None, cv2.INTER_LINEAR)
    return new_img

if __name__ == '__main__':
    # flows, w = FlowSplit(np.array([[[1.7,1.2]]]))
    # print(flows[0][0], w)
    import h5py
    root_dir = r'F:\datasets\MegVSR\train_png\84.mkv_down4x.mp4_frames'
    files = [os.path.join(root_dir, name) for name in os.listdir(root_dir)]
    # imgs = get_frames_from_video(85)
    # for k in range(2,20):
    #     flows = []
    #     images = []
    #     for i, file in enumerate(files[k:k+3]):
    #         images.append(cv2.imread(file)[:,:,::-1])
    #     for i in range(2):
    #         flow = calOptflow(images[i], images[2])
    #         flows.append(flow)
    #         visualize(flow, f'frame {i+1} --> frame 3')
    #     flowsplit, weights = FlowSplit(flows[1])
    #     new_flow = np.zeros_like(flows[1])
    #     # for i in range(2):
    #     #     for j in range(2):
    #     #         # new_flow += flowsplit[i][j]*weights[i][j]
    #     #         new_img = FlowShift(images[1], flowsplit[i][j])
    #     #         visualize(new_flow-flows[-1], f'new_flow')
    #     new_img = FlowShift(images[1], flows[1])
    #     cv2.imshow('new_img',images[1][:,:,::-1])
    #     # visualize(new_img, f'new_flow')
    #     # flow_add = calOptflow(images[0], images[1]) + calOptflow(images[1], images[2])
    #     # visualize(flow_add, f'frame 1 + frame 2 --> frame 3')
    #     # flow_sub = flow_add - flows[0]
    #     # visualize(flow_sub, f'frame 1+2->3  -  frame 1->3')
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    prvs_img = cv2.imread(files[0])[:,:,::-1]
    cv2.imshow('new_frames',prvs_img[:,:,::-1])
    cv2.imshow('flow',np.zeros_like(prvs_img))
    cv2.waitKey(1000)
    for i, file in enumerate(files[1:]):
        next_img = cv2.imread(file)[:,:,::-1]
        flow = calOptflow(prvs_img, next_img, True)
        prvs_img = next_img
    #     flows.append(flow)
    # flows.append(np.zeros_like(flow))
    # flows = np.array(flows).astype(np.float32)
    # f = h5py.File('flows.h5', 'w')
    # dst_hr = f.create_dataset('85', data=flows)
    # f.close()

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

