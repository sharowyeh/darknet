"""
Represent demo module from src/demo.c
Functions given camera index or media file for frame detection
"""

import ctypes
from darknet_libwrapper import *
import cv2
import numpy as np

alphabet = load_alphabet()

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(ctypes.c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def ipl_to_image(ipl):
    im = array_to_image(ipl)
    rgbgr_image(im)
    return im

def image_to_ipl(im):
    rgbgr_image(im)
    # super fast!!
    buff = np.ctypeslib.as_array(im.data, shape=(im.c, im.h, im.w))
    buff = (buff * 255.0).astype(np.uint8)
    buff = buff.transpose(1,2,0)
    # following steps cost resource transform ctypes.POINTER to ndarray
    #ptr = ctypes.cast(im.data, ctypes.POINTER(ctypes.c_float * (im.c * im.h * im.w)))
    #buff = np.fromiter(ptr.contents, dtype=np.float, count=(im.c * im.h * im.w))
    #buff = (buff * 255.0).reshape(im.c, im.h, im.w).astype(np.uint8)
    #buff = buff.transpose(1,2,0)
    return buff

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
                # Begins from yolov3, bbox coords has been changed to % of image's width and height
                # via network_predict(), remember_network() and avg_predictions()
                # TODO: solve this workaround for coords via get_network_boxes()
                dets[j].bbox.x = b.x / image.w
                dets[j].bbox.w = b.w / image.w
                dets[j].bbox.y = b.y / image.h
                dets[j].bbox.h = b.h / image.h
    res = sorted(res, key=lambda x: -x[1])
    draw_detections(image, dets, num, thresh, meta.names, alphabet, meta.classes)
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
    print('cap is open?', cap.isOpened(), width, height)
    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Demo', int(width), int(height))
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        im = ipl_to_image(frame)
        result = _detector(net, meta, im)
        disp = image_to_ipl(im)
        cv2.imshow('Demo', disp)
        key = cv2.waitKey(1)
        if key == 27:
            break
        #print('result:', result)
    cv2.destroyAllWindows()
    cap.release()
