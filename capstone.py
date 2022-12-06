import RPi.GPIO as GPIO  # 파이썬의 GPIO 모듈 불러온다.
import time  # 파이썬에서 time 모듈을 불러온다.
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import pymysql
import pygame
import schedule

led_r = 13
led_g = 19

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(led_r, GPIO.OUT)
GPIO.setup(led_g, GPIO.OUT)

date = datetime.datetime.now() #날짜 정보 가져오기

#경고음 출력
pygame.mixer.init()
p = pygame.mixer.Sound("sam.wav")


GPIO.setmode(GPIO.BCM)  # GPIO의 라즈베리파이 GPIO번호로 사용

TRIG = 10  # 핀번호 설정
ECHO = 9  # 핀번호 설정

GPIO.setup(TRIG, GPIO.OUT)  # 초음파 출력
GPIO.setup(ECHO, GPIO.IN)  # 초음파 입력

VideoSignal = cv2.VideoCapture(0) #웹캡 신호 받기

GPIO.output(led_g, GPIO.LOW)
GPIO.output(led_r, GPIO.LOW)

# try내에 반복문을 이용해 송수신 설정 및 거리 측정
def parking_protector():
    try:
        while True:

            GPIO.output(TRIG, False)
            time.sleep(0.5)

            GPIO.output(TRIG, True)
            time.sleep(0.00001)
            GPIO.output(TRIG, False)

            while GPIO.input(ECHO) == 0:
                start = time.time()

            while GPIO.input(ECHO) == 1:
                end = time.time()

            check_time = end - start
            distance = check_time * 17000
            print("거리 : %.1f cm" % distance)
            # 거리 = distance
            if (distance <= 200):
                ret, image = VideoSignal.read()
                height, width, center = image.shape
                fps = 30
                cn= 1

                fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                out = cv2.VideoWriter('result/webcam.avi', fcc, fps, (width, height))

                ret, frame = VideoSignal.read()
                if ret:
                    out.write(frame)
                    # cv2.imshow('frame', frame)
                    cv2.imwrite('result/screenshot{}.png'.format(cn), frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

                image = cv2.imread('result/screenshot{}.png'.format(cn))

                image_resize = cv2.resize(image, None, fx=0.4, fy=0.4)

                # YOLO 가중치 파일과 CFG 파일 로드
                YOLO_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
                # YOLO NETWORK 재구성
                classes = []
                with open("coco.names", "r") as f:
                    classes = [line.strip() for line in f.readlines()]
                layer_names = YOLO_net.getLayerNames()

                output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

                # 웹캠 프레임
                h, w, c = image_resize.shape

                # YOLO 입력
                blob = cv2.dnn.blobFromImage(image_resize, 0.00392, (416, 416), (0, 0, 0),
                                             True, crop=False)
                YOLO_net.setInput(blob)
                outs = YOLO_net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * w)
                            center_y = int(detection[1] * h)
                            dw = int(detection[2] * w)
                            dh = int(detection[3] * h)
                            # Rectangle coordinate
                            x = int(center_x - dw / 2)
                            y = int(center_y - dh / 2)
                            boxes.append([x, y, dw, dh])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

                for i in range(len(boxes)):

                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        score = confidences[i]

                        # 경계상자와 클래스 정보 이미지에 입력
                        cv2.rectangle(image_resize, (x, y), (x + w, y + h), (0, 0, 255), 5)
                        cv2.putText(image_resize, label, (x, y - 20), cv2.FONT_ITALIC, 1,
                                    (255, 255, 255), 1)

                if (label == 'car'):

                    plt.style.use('dark_background')

                    # 이미지 불러오기
                    img_ori = image

                    height, width, channel = img_ori.shape

                    # 이미지에서 글씨를 읽기 쉽게 하기 위해 그레이로 변경
                    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

                    # 밝은 부분과 어두운 부분를 명확하게 구분하기 위해 콘트라스트를 변경
                    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

                    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
                    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

                    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
                    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

                    # 가우시안블러 처리
                    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

                    # 이미지 흑백화
                    img_thresh = cv2.adaptiveThreshold(
                        img_blurred,
                        maxValue=255.0,
                        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        thresholdType=cv2.THRESH_BINARY_INV,
                        blockSize=19,
                        C=9
                    )

                    # 윤곽선 찾기
                    contours, _ = cv2.findContours(
                        img_thresh,
                        mode=cv2.RETR_LIST,
                        method=cv2.CHAIN_APPROX_SIMPLE
                    )

                    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

                    # Prepare Data
                    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                    contours_dict = []

                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

                        # insert to dict
                        contours_dict.append({
                            'contour': contour,
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'cx': x + (w / 2),
                            'cy': y + (h / 2)
                        })

                    # select Candidates by char Size
                    MIN_AREA = 80
                    MIN_WIDTH, MIN_HEIGHT = 2, 8
                    MIN_RATIO, MAX_RATIO = 0.25, 1.0

                    possible_contours = []

                    cnt = 0
                    for d in contours_dict:
                        area = d['w'] * d['h']
                        ratio = d['w'] / d['h']

                        if area > MIN_AREA \
                                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                                and MIN_RATIO < ratio < MAX_RATIO:
                            d['idx'] = cnt
                            cnt += 1
                            possible_contours.append(d)

                    # visualize possible contours
                    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                    for d in possible_contours:
                        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
                        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                                      color=(255, 255, 255),
                                      thickness=2)

                    # Select Candidates by Arrangement of Contours
                    MAX_DIAG_MULTIPLYER = 5  # 5
                    MAX_ANGLE_DIFF = 12.0  # 12.0
                    MAX_AREA_DIFF = 0.5  # 0.5A
                    MAX_WIDTH_DIFF = 0.8
                    MAX_HEIGHT_DIFF = 0.2
                    MIN_N_MATCHED = 3  # 3

                    def find_chars(contour_list):
                        matched_result_idx = []

                        for d1 in contour_list:
                            matched_contours_idx = []
                            for d2 in contour_list:
                                if d1['idx'] == d2['idx']:
                                    continue

                                dx = abs(d1['cx'] - d2['cx'])
                                dy = abs(d1['cy'] - d2['cy'])

                                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                                distance = np.linalg.norm(
                                    np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                                if dx == 0:
                                    angle_diff = 90
                                else:
                                    angle_diff = np.degrees(np.arctan(dy / dx))
                                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                                    matched_contours_idx.append(d2['idx'])

                            # append this contour
                            matched_contours_idx.append(d1['idx'])

                            if len(matched_contours_idx) < MIN_N_MATCHED:
                                continue

                            matched_result_idx.append(matched_contours_idx)

                            unmatched_contour_idx = []
                            for d4 in contour_list:
                                if d4['idx'] not in matched_contours_idx:
                                    unmatched_contour_idx.append(d4['idx'])

                            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

                            # recursive
                            recursive_contour_list = find_chars(unmatched_contour)

                            for idx in recursive_contour_list:
                                matched_result_idx.append(idx)

                            break

                        return matched_result_idx

                    result_idx = find_chars(possible_contours)

                    matched_result = []
                    for idx_list in result_idx:
                        matched_result.append(np.take(possible_contours, idx_list))

                    # visualize possible contours
                    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                    for r in matched_result:
                        for d in r:
                            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                                          color=(255, 255, 255),
                                          thickness=2)

                    # Rotate Plate Images
                    PLATE_WIDTH_PADDING = 1.3  # 1.3
                    PLATE_HEIGHT_PADDING = 1.5  # 1.5
                    MIN_PLATE_RATIO = 3
                    MAX_PLATE_RATIO = 10

                    plate_imgs = []
                    plate_infos = []

                    for i, matched_chars in enumerate(matched_result):
                        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

                        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
                        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

                        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0][
                            'x']) * PLATE_WIDTH_PADDING

                        sum_height = 0
                        for d in sorted_chars:
                            sum_height += d['h']

                        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

                        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
                        triangle_hypotenus = np.linalg.norm(
                            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
                            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
                        )

                        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

                        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

                        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

                        img_cropped = cv2.getRectSubPix(
                            img_rotated,
                            patchSize=(int(plate_width), int(plate_height)),
                            center=(int(plate_cx), int(plate_cy))
                        )

                        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / \
                                img_cropped.shape[
                                    0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                            continue

                        plate_imgs.append(img_cropped)
                        plate_infos.append({
                            'x': int(plate_cx - plate_width / 2),
                            'y': int(plate_cy - plate_height / 2),
                            'w': int(plate_width),
                            'h': int(plate_height)
                        })

                    longest_idx, longest_text = -1, 0
                    plate_chars = []

                    for i, plate_img in enumerate(plate_imgs):
                        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
                        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0,
                                                     type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                        # find contours again (same as above)
                        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

                        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
                        plate_max_x, plate_max_y = 0, 0

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)

                            area = w * h
                            ratio = w / h

                            if area > MIN_AREA \
                                    and w > MIN_WIDTH and h > MIN_HEIGHT \
                                    and MIN_RATIO < ratio < MAX_RATIO:
                                if x < plate_min_x:
                                    plate_min_x = x
                                if y < plate_min_y:
                                    plate_min_y = y
                                if x + w > plate_max_x:
                                    plate_max_x = x + w
                                if y + h > plate_max_y:
                                    plate_max_y = y + h

                        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

                        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
                        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0,
                                                      type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10,
                                                        borderType=cv2.BORDER_CONSTANT,
                                                        value=(0, 0, 0))

                        # chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
                        chars = pytesseract.image_to_string(img_result, lang='kor')
                        result_chars = ''
                        has_digit = False
                        for c in chars:
                            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                                if c.isdigit():
                                    has_digit = True
                                result_chars += c

                        # print(result_chars)
                        plate_chars.append(result_chars)

                        if has_digit and len(result_chars) > longest_text:
                            longest_idx = i

                        info = plate_infos[longest_idx]
                        chars = plate_chars[longest_idx]

                        print(chars)
                        img_out = img_ori.copy()

                        cv2.rectangle(img_out, pt1=(info['x'], info['y']),
                                      pt2=(info['x'] + info['w'], info['y'] + info['h']),
                                      color=(255, 0, 0), thickness=2)

                        plt.imshow(img_result)
                        plt.imsave(chars + '.jpg', img_out)
                        plt.figure(figsize=(12, 10))
                        plt.imshow(img_out)
                        plt.show()
                        # 번호 저장 -> chars
                        # mysql 접속
                        conn = pymysql.connect(host='localhost',
                                               user='root',
                                               password='1234',
                                               db='parking_protecter',
                                               charset='utf8')

                        with conn:
                            with conn.cursor() as cur:
                                try:
                                    # sql 생성문 선언
                                    sql = f"SELECT car_number FROM exam WHERE car_number = '{chars}'"
                                    cur.execute(sql)
                                    print(cur.execute(sql))
                                    if 1 > cur.execute(sql):  # 일치하는 번호가 없으면
                                        print('일치하는 번호가 없습니다.')
                                        GPIO.output(led_r, GPIO.HIGH)
                                        GPIO.output(led_g, GPIO.LOW)
                                        p.play()
                                        time.sleep(10)

                                    else:
                                        print('등록된 번호 입니다.')
                                        GPIO.output(led_g, GPIO.HIGH)
                                        GPIO.output(led_r, GPIO.LOW)
                                        time.sleep(10)


                                except:x
                                    print('에러가 발생하였습니다.')
                else:
                    print("차량이 아닙니다.")
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                    cn += 1

    except KeyboardInterrupt:
        print("거리 측정 완료")
        GPIO.cleanup()

schedule.every(5).minute.do(parking_protector()) #5분마다 실행

while True:
    schedule.run_pending()
    time.sleep(1)