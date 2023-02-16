import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single, get_hand_location, draw_single_original_image

from Track.Tracker import Detection, Tracker
#from ActionsEstLoader import TSSTG

import mysql.connector

#source = '../Data/test_video/test7.mp4'
#source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
source = '../Data/falldata/Home/Videos/video (1).avi'
#source = 2
patient_area_1 = (253, 206), (433, 245)
patient_area_2 = (136, 348), (445, 435)


def db_connection():
    try:
        mydb = mysql.connector.connect(
            host="192.168.1.103",
            user="root",
            port="3306",
            database="QE",
            passwd="edgeimcl",
            autocommit=True)
        return mydb
    except mysql.connector.Error as err:
      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
      elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
      else:
        print(err)
    else:
        return mydb

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''
    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[1][0]) - np.maximum(a[0], b[0][0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[1][1]) - np.maximum(a[1], b[0][1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[1][0] - b[0][0])*(b[1][1] - b[0][1])
  
    return float(s_intsec)/(s_a + s_b -s_intsec)



if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    # action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc, ori_return=True).start()

    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        # codec = cv2.VideoWriter_fourcc(*'MJPG')
        # writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_out, codec, 5, (inp_dets, inp_dets))


    fps_time = 0
    f = 0
    mydb = db_connection()
    cursor = mydb.cursor()
    id_inserted = []

    while cam.grabbed():
        f += 1
        frame, image = cam.getitem()
        #image = frame.copy()
        #f += 1
        #if f % 5 != 0:
            #frame = cam.getitem()
            #continue

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
        #if f == 5:
            #print(frame.shape)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        new_detected = []
        new_detected_scores = []
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections, frame)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            # track_role = track.role
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            image_bbox = [bbox[0]*2, (bbox[1]-140)*2, bbox[2]*2, (bbox[3]-140)*2]
            image_center = [center[0]*2, (center[1]-140)*2]


            #action = 'pending..'
            #clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            #if len(track.keypoints_list) == 30:
                #pts = np.array(track.keypoints_list, dtype=np.float32)
                #out = action_model.predict(pts, frame.shape[:2])
                #action_name = action_model.class_names[out[0].argmax()]
                #action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                #if action_name == 'Fall Down':
                    #clr = (255, 0, 0)
                #elif action_name == 'Lying Down':
                    #clr = (255, 200, 0)


            # VISUALIZE.
            if track.time_since_update == 0:
                    if args.show_skeleton:
                        image = draw_single_original_image(image, track.keypoints_list[-1])

                    if center[1] < 320:
                        # in bed-1 area
                        if box_iou2(bbox,patient_area_1) < 0.3:
                            action = 'leaving bed'
                            clr = (255, 0, 0)
                            # insert data to database
                            if track_id not in id_inserted:
                                info = "insert into leavebed (time, bed, status) values ({}, {}, '{}')".format('NOW()', '1', action)
                                cursor.execute(info)
                                id_inserted.append(track_id)
                        else:
                            action = 'on bed'
                            clr = (255, 200, 0)
                            #insert data to database
                            info = "insert into leavebed (time, bed, status) values ({}, {}, '{}')".format('NOW()' '1', action)
                            cursor.execute(info)
                    else:
                        # in bed-2 area
                        if box_iou2(bbox,patient_area_2) < 0.3:
                            action = 'leaving bed'
                            clr = (255, 0, 0)
                            # insert data to database
                            if track_id not in id_inserted:
                                info = "insert into leavebed (time, bed, status) values ({}, {}, '{}')".format('NOW()', '2', action)
                                cursor.execute(info)
                                id_inserted.append(track_id)
                        else:
                            action = 'on bed'
                            clr = (255, 200, 0)
                            #insert data to database
                            info = "insert into leavebed (time, bed, status) values ({}, {}, '{}')".format('NOW()', '2', action)
                            cursor.execute(info)


                    image = cv2.rectangle(image, (image_bbox[0], image_bbox[1]), (image_bbox[2], image_bbox[3]), (0, 255, 0), 1)
                    image = cv2.putText(image, str(track_id), (image_center[0], image_center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
                    image = cv2.putText(image, action, (image_bbox[0] + 5, image_bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, clr, 2)
                    # image = cv2.putText(image, track_role, (image_center[0]+20, image_center[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
     
        # Show Frame.
        # frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        # draw patient area yellow
        image_patient_area_1 = (patient_area_1[0][0]*2, (patient_area_1[0][1]-140)*2),(patient_area_1[1][0]*2, (patient_area_1[1][1]-140)*2)
        image_patient_area_2 = (patient_area_2[0][0]*2, (patient_area_2[0][1]-140)*2),(patient_area_2[1][0]*2, (patient_area_2[1][1]-140)*2)

        image = cv2.rectangle(image, image_patient_area_1[0], image_patient_area_1[1], (0,255,255), 2)
        image = cv2.rectangle(image, image_patient_area_2[0], image_patient_area_2[1], (0,255,255), 2)


        #frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            #(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        
        image = image[:, :, ::-1]
        #fps_time = time.time()
        #if f == 150 or f == 200 or f == 250:
        #if cv2.waitKey(1) & 0xFF == ord('s'):
        #if f == 5:
            #print(frame.shape)
            #cv2.imwrite(str(f) + '.jpg', frame)

        if outvid:
            writer.write(image)

        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
