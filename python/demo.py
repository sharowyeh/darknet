"""
Represent demo module from src/demo.c
Functions given camera index or media file for frame detection
"""

import ctypes
from darknet_libwrapper import *
import cv2

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(ctypes.c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def _detector(net, meta, image, thresh=.5, hier=.5, nms=.45):
    cuda_set_device(0)
    num = ctypes.c_int(0)
    num_ptr = ctypes.pointer(num)
    network_predict_image(net, image)
    dets = get_network_boxes(net, image.w, image.h, thresh, hier, None, 0, num_ptr)
    num = num_ptr[0]
    if (nms):
         do_nms_sort(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                # Notice: in Python3, mata.names[i] is bytes array from c_char_p instead of string
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res

def _demo(*argv):
    """argv: 'darknet' 'XXXX' data cfg weight thresh cam_index mp4 class_names class_num frame_skip prefix frame_avg hier w h fps full_screen"""
    """
    Call to lib export demo
    NOTICE: data loaded at XXXX.c
    """
    print('Not implement')

def demo(*argv):
    """argv: 'darknet' 'XXXX' data cfg weight thresh cam_index mp4 class_names class_num frame_skip prefix frame_avg hier w h fps full_screen"""
    print('demo:', 'data:{2} cfg:{3} weight:{4} cam:{6} video:{7}'.format(*argv))
    meta = get_metadata(argv[2])
    net = load_network(argv[3], argv[4], 0)
    set_batch_network(net, 1)
    cam_index = argv[6]
    video_file = argv[7]
    if video_file is not None:
        cap = cv2.VideoCapture(video_file)
    else:
        cap = cv2.VideoCapture(cam_index)
    # CV_CAP_PROP_FRAME_WIDTH = 3
    width = cap.get(3)
    # CV_CAP_PROP_FRAME_HEIGHT = 4
    height = cap.get(4)
    # CV_CAP_PROP_FPS = 5
    fps = cap.get(5)
    print('cap is open?', cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        im = array_to_image(frame)
        rgbgr_image(im)
        result = _detector(net, meta, im)
        print('result:', result)
    cap.release()
