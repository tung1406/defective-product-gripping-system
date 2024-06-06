import os
import time
from pathlib import Path
from PIL import Image
import datetime
import mysql.connector
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import argparse

from models.experimental import attempt_load
# from utils.plots import Annotator, colors
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, set_logging, increment_path, strip_optimizer

from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# ------------ Import library qt5--------------------

import cv2
# from queue import Queue
# from multiprocessing import Queue

import base64

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

import serial


serialPort = serial.Serial(port="COM4", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)


status_send = 0

import numpy as np
object_counter = {}
line_track_id = []

Only = True

Id_time = None
day = None
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
confident = {}

Camera_main_loop = True

q_count_height = None
q_track_height = None
confident_track = 0
Hour = None
RESETBUTTON = False
DIR_path = None
directory = None
SUM_pre = 0
DATCHUAN_pre = 0
LOI1_pre = 0
LOI2_pre = 0
SUM = 0
DATCHUAN = 0
LOI1 = 0
LOI2 = 0
resized_img = None
save_path = None

obj_name = None
coordinates = [] #danh sach luu tru toa do

# mydb = mysql.connector.connect(host='localhost', user='root', password='')
# mycursor = mydb.cursor()

def track_up(q_track_1, q_track_2, Camera_main_loop_1, Reset):
    global q_count_height
    global q_track_height
    global Camera_main_loop
    global RESETBUTTON
    # global object_counter
    # global data_deque
    global object_counter
    q_count_height = q_track_1
    q_track_height = q_track_2
    Camera_main_loop = Camera_main_loop_1
    RESETBUTTON = Reset
    if RESETBUTTON is True:
        object_counter = {}
        # print("Object counter Is True")

def xyxy_to_xywh(*xyxy):

    # Calculates the relative bounding box from absolute pixel values
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):

    # Simple function that adds fixed color depending on the class
    if label == 0:  #
        color = (85, 45, 255)
    elif label == 1:  #
        color = (222, 82, 175)
    elif label == 2:  #
        color = (0, 204, 255)
    elif label == 5:  #
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

# draw labels khung chu nhat ben tren class dder khi khung chu nhat do di qua line chung ta tang count len 1
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # ve hinh chu nhat quanh doi tuong
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # ve border cho class text cuar yolo
        # img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)

def intersect(A, B, C, D):
    # data_deque[id][0], data_deque[id][1], line[0], line[1]
    # x1 + x2, y1
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw_track(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect_track(A, B, C, D):
    # data_deque[id][0], data_deque[id][1], line[0], line[1]
    # x1 + x2, y1
    return ccw_track(A, C, D) != ccw_track(B, C, D) and ccw_track(A, B, C) != ccw_track(A, B, D)


def trackpop(A, B):
    return A < B

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def get_direction(point1, point2):
    direction_str = ""
    # huong nam la huong ma y1 > y2
    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        # huong bac la huong y2 > y1
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        # print(f'label: {label}') a
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def time_now():
    global day
    global Id_time
    global Hour

    now = datetime.datetime.now()
    day = now.strftime("%Y_%m_%d")  # %m/%d/%Y
    Hour = now.hour
    Id_time = now.strftime("%H:%M:%S")  # to sql Time

    return day

def save_img_Spl(image, ID_track, obj_name, x1, y1, x2, y2):

    if "loi" in obj_name:
        box_image = image[y1:y2, x1:x2]

        global resized_img
        global encoded_img

        # Tăng kích thước của ảnh bằng phương pháp interpolate
        scale_percent = 400  # Tăng kích thước ảnh lên gấp đôi
        width = int(box_image.shape[1] * scale_percent / 100)
        height = int(box_image.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_img = cv2.resize(box_image, dim, interpolation=cv2.INTER_LINEAR)

        # Convert image-array to BytesIO
        image_buffer = cv2.imencode('.jpg',resized_img, None)[1].tobytes()

        # Convert BytesIO to Base64_string
        encoded_img = base64.b64encode(image_buffer).decode("utf-8")

    # Write_MySql(ID_track, obj_name)

def inint_Mysql():
    global mydb
    global mycursor

    # Thực thi truy vấn SQL để liệt kê tất cả các cơ sở dữ liệu
    mycursor.execute("SHOW DATABASES")

    # Lấy kết quả từ truy vấn
    databases = mycursor.fetchall()
    if ('mydatabase',) not in databases:
        mycursor.execute("CREATE DATABASE mydatabase")

def Write_MySql(ID_track, obj_name):
    time_now()
    global day
    global save_path
    global mydb
    global object_counter
    global SUM_pre
    global DATCHUAN_pre
    global LOI1_pre
    global LOI2_pre
    global SUM
    global DATCHUAN
    global LOI1
    global LOI2
    global Id_time
    global Hour
    global DIR_path
    global resized_img
    global save_path
    global encoded_img

    Time = Id_time
    Table_name = day
    table_data = 'mydata'   # name of table-data

    # inint_Mysql()
    mycursor.execute("USE mydatabase")
    mycursor.execute("SHOW TABLES")

    # Lấy kết quả từ truy vấn
    tables = mycursor.fetchall()

    if (f'{table_data}',) not in tables:
        mycursor.execute(
            f"CREATE TABLE {table_data} ( id INT AUTO_INCREMENT PRIMARY KEY, error VARCHAR(255), Dayy VARCHAR(255), Timee VARCHAR(255), img_base64 LONGTEXT, ID_TRACK INT , SUM  INT, DATCHUAN INT, LOI1  INT, LOI2 INT)")

    if 'dat chuan' in object_counter:
        DATCHUAN = object_counter['dat chuan']
    if 'loi 1' in object_counter:
        LOI1 = object_counter['loi 1']
    if 'loi 2' in object_counter:
        LOI2 = object_counter['loi 2']
    SUM = DATCHUAN + LOI1 + LOI2

    if "loi" in obj_name:
        sql = f"INSERT INTO {table_data} (error, Dayy, Timee, img_base64, ID_TRACK, SUM, DATCHUAN, LOI1, LOI2) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (f"{obj_name}", f"{day}", f"{Time}", f"{encoded_img}", f"{ID_track}", f"{SUM}", f"{DATCHUAN}", f"{LOI1}",
                   f"{LOI2}")
        mycursor.execute(sql, val)
    else:
        sql = f"INSERT INTO {table_data} (error, Dayy, Timee, ID_TRACK, SUM, DATCHUAN, LOI1, LOI2) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (f"{obj_name}", f"{day}", f"{Time}", f"{ID_track}", f"{SUM}", f"{DATCHUAN}", f"{LOI1}",
               f"{LOI2}")
        mycursor.execute(sql, val)

    mydb.commit()

def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0), confs=[]):
    global line
    global confident
    global confident_track
    global line_track_id
    global q_count_height
    global obj_name

    _, width, _ = img.shape
    if q_count_height is None:
        height, width, _ = img.shape
    elif q_count_height is 0:
        height, width, _ = img.shape
    else:
        height = q_count_height * 7 / 3
        #print("thuid", identities)
    start_point = (0, int(height / 7 * 3 +16))  # height- 100
    end_point = (width, int(height / 7 * 3 +16))
    start_point_1 = (0, int(height / 7 * 3))  # height- 100
    end_point_1 = (width, int(height / 7 * 3))

    line = [start_point_1, end_point_1]
    #print("height: ", height)
    cv2.line(img, start_point, end_point, (46, 162, 112), 3)

    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)
            if key not in confident:
                confident.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]

        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
        center_top = (int((x2 + x1) / 2), int(y1))
        center_compare = int(y1)

        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        color = compute_color_for_labels(object_id[i])

        if (confident_track < len(confs) - 1):
            confident_track = confident_track + 1
        elif (confident_track >= len(confs) - 1):
            confident_track = 0

        if (len(confs) == 1):
            confident_track = 0

        if (len(confs) > 0):
            confident[id] = confs[confident_track][0]

        # write parameter for label box
        label = '{}{:d}'.format("", id) + ":" + '%s' % (names[object_id[i]]) + ":{:.2f}".format(confident[id])

        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:

            direction = get_direction(data_deque[id][0], data_deque[id][1])
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, start_point, end_point, (255, 255, 255), 3)

                # Counting
                if "South" in direction:
                    # "West" in direction
                    if names[object_id[i]] not in object_counter:
                        object_counter[names[object_id[i]]] = 1

                    else:
                        object_counter[names[object_id[i]]] += 1

                #print(names[object_id[i]])

                # gán obj_name đã được đếm
                obj_name = names[object_id[i]]
                save_img_Spl(img, id, obj_name, x1, y1, x2, y2)

        UI_box(box, img, label=label, color=color, line_thickness=1)

    return img


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


# def detect(q_img_origin:Queue,q1_img_track:Queue,q2_count_error:Queue,opt, show_vid=False,tkinter_is=False, StopIterationq=None):
def detect(q_img_origin, q1_img_track, q2_count_error, opt):
    global status_send
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    global t11
    global t00
    global cout

    global q_track_height
    global Camera_main_loop
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # initialize deepsort
    cfg_deep = get_config()

    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    cv2.waitKey(1)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage: classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadWebcam(source, img_size=imgsz, stride=stride)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        print('da tai xong loadSteam')
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()
    print("Start Tracking..")
    for path, img, im0s, vid_cap in dataset:
        t00 = time_synchronized()
        if Camera_main_loop is True:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, sToAddString, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, sToAddString, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                w, h = im0.shape[1], im0.shape[0]
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt

                # Camera input
                q_img_origin.put(im0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        sToAddString += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # height, width, _ = im0.shape
                    _, width, _ = im0.shape
                    # height = 476 * 7
                    if q_track_height is None:
                        height, width, _ = im0.shape
                    elif q_track_height is 0:
                        height, width, _ = im0.shape
                    else:
                        height = q_track_height * 7
                        # height = 476*7
                    # height = 476 * 7
                    y_track_pop = int(height / 7)
                    # 100  # height - 400
                    start_point_track = (0, int(height / 7))
                    end_point_track = (width, int(height / 7))
                    line_track_id = [start_point_track, end_point_track]

                    start_point_grab = (0, int(height/7 * 2))  # height = 720, y = 200
                    end_point_grab = (width, int(height /7 * 2))


                    xywh_bboxs = []
                    confs = []
                    oids = []
                    outputs = []

                    for *xyxy, conf, cls in reversed(det):

                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)

                        cv2.line(im0, line_track_id[0], line_track_id[1], (128, 0, 128), 3)



                        center_top = (int((x_c + bbox_w) / 2), int(y_c))
                        center_compare = int(y_c)

                        if trackpop(center_compare, y_track_pop):
                            continue
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])
                        oids.append(int(cls))
                        #print(cls)

                        if int(cls) == 1 and status_send == 0:
                            cv2.line(im0, start_point_grab, end_point_grab, (232, 247, 11), 3)
                            cv2.putText(im0, "loi muc", (int(x_c),int(y_c)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
                            if int(y_c) > int(int(height/7 * 2 - 5)) and int(y_c) < int(int(height/7 * 2 + 20)):
                                    cv2.line(im0, start_point_grab, end_point_grab, (255, 255, 255), 3)
                                    print(x_c)
                                    cv2.putText(im0, "target", (int(x_c), int(y_c)), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                                            (0, 0, 255), 2, cv2.LINE_AA)

                                    datasend = str(x_c) + "\n"
                                    serialPort.write(datasend.encode())
                                    status_send = 1

                        if serialPort.in_waiting > 0:
                            serialString = serialPort.readline()  # b'done'
                            serialString = str(serialString.decode('utf8').replace("b'\n", ""))
                            if serialString == 'done':
                                status_send = 0

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    outputs = deepsort.update(xywhs, confss, oids, im0)

                    # Trong vòng lặp xác định tọa độ
                    #for *xyxy, conf, cls in reversed(det):
                    #    # ... Xác định tọa độ ...
                    #    coordinates.append((x_c, y_c))  # Thêm tọa độ vào danh sách

                    #if names[object_id[i]] != 'dat chuan':
                    #    for coord in coordinates:
                    #        x, y = coord
                    #        print(f"Object coordinates: ({x}, {y})")



                    if len(outputs) > 0:
                        # for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bbox_xyxy = outputs[:, :4]

                        identities = outputs[:, -2]
                        object_id = outputs[:, -1]


                        draw_boxes(im0, bbox_xyxy, names, object_id, identities, offset=(0, 0), confs=confs)

                t4 = time_synchronized()

                # Tinh FPS
                fps_exc = 1 / (t4 - t00)  # FPS for 1 frame
                # fps_deepsort = "FPS: " + str(round(fps_exc, 1))
                # cv2.line(im0, (20, 25), (180, 25), [85, 45, 255], 40)
                # cv2.putText(im0, fps_deepsort, (11, 35), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # print(
                #     f'Done per frame yolov7 + tracking. ({(1E3 * (t4 - t00)):.1f}ms),   FPS = {fps_exc:.0f} (Frame/s),  Inference ({(1E3 * (t2 - t1)):.1f}ms),  NMS ({(1E3 * (t3 - t2)):.1f}ms)')

                height, width, _ = im0.shape
                q1_img_track.put(im0)
                q2_count_error.put(object_counter)

                if view_img:
                    # tu them yolov5
                    color = (0, 255, 0)

                    start_point = (0, h - 100)
                    end_point = (w, h - 100)

                    thickness = 3
                    org = (150, 150)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 3

                # Save result
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                # Debug
                                print("...")
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                        vid_writer.write(im0)

    print(f'Done video. ({time.time() - t0:.3f}s)')
