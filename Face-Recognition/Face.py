# coding=utf-8
import os
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
from MultiMsgSync import TwoStageHostSeqSync


parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])#array of length bbox filled with value=frame.shape[0]

    normVals[::2] = frame.shape[1] #values at 0,2,4,6....indexes=frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int) #returns normVals array but negative values are clipped to 0 and values >=1 are kept intact.


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

VIDEO_SIZE = (1072, 1072)
databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)

#Putting text in the frame
class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)#black
        self.color = (255, 255, 255)#white
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX #font style
        self.line_type = cv2.LINE_AA #type of line
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type) #font scale=1 and thickness =4
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)

class FaceRecognition:
    def __init__(self, db_path, name) -> None:
        self.read_db(db_path)
        self.name = name
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True

    def cosine_distance(self, a, b): #cosine returns similarity between two vectors
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)   #modulus of vecotr a
        b_norm = np.linalg.norm(b)   #modulus of vecotr b
        return np.dot(a, b.T) / (a_norm * b_norm)   #cos(theta)= A.B/mod(A).mod(B)
        #cosine value higher means angle smaller i.e similarity high!!!

    def new_recognition(self, results): #will return name array of the most matched person with value: i.e name=[disimilarity,name]
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels): #in read_db method ,there is  self.labels = []
            for j in self.db_dic.get(label): #db_dic : dictionary of person data with name as label and data array as value
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_)) #max_=max similarity between result and a data with label as person name
        
        #if similarity is >=0.5 ,then the face is of person name="{label_}"
        #1 - conf[0][0] is dis-similarity between result and a data with label as person name.


        # self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        # self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))
        nameText=''
        if conf[0][0]<0.5:
            self.create_db(results)
            nameText=self.name
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0],nameText)
        return name

    def read_db(self, databases_path):
        self.labels = []
        
        for folder in os.listdir(databases_path):
            file=os.listdir(f"{databases_path}/{folder}")[0]
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0]) #getting the names from the database

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases_path}/{label}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files] #saving database as key:person name and val:array of all the datas.

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    def create_db(self, results):
        
        # if self.name is None:
        #     if not self.printed:
        #         print("Wanted to create new DB for this face, but --name wasn't specified")
        #         self.printed = True
        #     return
        if self.name is None:
            count=0
            if os.path.exists(databases):
                count=len(os.listdir(databases))+1
                #count=len(databases)+1
            self.name='unknown'+str(count)
            #self.name='unknown'+str(number)
        print('Saving face...')
        try:
            with np.load(f"{databases}/{self.name}/{self.name}.npz") as db:
                #db.file
                db_ = [db[j] for j in db.files][:] #db[i][:] all contents of ith row
        except Exception as e:
            db_ = []
        db_.append(np.array(results))

        #database---->name
        #             ---name.png
        #             ---name.npz
        if not os.path.exists(f"{databases}/{self.name}"):
            os.mkdir(f"{databases}/{self.name}")
        

        np.savez_compressed(f"{databases}/{self.name}/{self.name}", *db_)
        self.adding_new = False
        

print("Creating pipeline...")
pipeline = dai.Pipeline()

print("Creating Color Camera...")
cam = pipeline.create(dai.node.ColorCamera)

# For ImageManip rotate you need input frame of multiple of 16
cam.setPreviewSize(1072, 1072)
cam.setVideoSize(VIDEO_SIZE)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

host_face_out = pipeline.create(dai.node.XLinkOut)
host_face_out.setStreamName('color')
cam.video.link(host_face_out.input)

# ImageManip as a workaround to have more frames in the pool.
# cam.preview can only have 4 frames in the pool before it will
# wait (freeze). Copying frames and setting ImageManip pool size to
# higher number will fix this issue.
copy_manip = pipeline.create(dai.node.ImageManip)
cam.preview.link(copy_manip.inputImage)
copy_manip.setNumFramesPool(20)
copy_manip.setMaxOutputFrameSize(1072*1072*3)

# ImageManip that will crop the frame before sending it to the Face detection NN node
face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
copy_manip.out.link(face_det_manip.inputImage)

# NeuralNetwork
print("Creating Face Detection Neural Network...")
face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))

face_det_manip.out.link(face_det_nn.input)

face_det_xout = pipeline.create(dai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'age_gender_manip' to crop the initial frame
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

face_det_nn.out.link(script.inputs['face_det_in'])
# We also interested in sequence number for syncing
face_det_nn.passthrough.link(script.inputs['face_pass'])

copy_manip.out.link(script.inputs['preview'])

with open("script.py", "r") as f:
    script.setScript(f.read())

print("Creating Head pose estimation NN")

headpose_manip = pipeline.create(dai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
headpose_manip.setWaitForConfigInput(True)
script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
script.outputs['manip_img'].link(headpose_manip.inputImage)

headpose_nn = pipeline.create(dai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
headpose_manip.out.link(headpose_nn.input)

headpose_nn.out.link(script.inputs['headpose_in'])
headpose_nn.passthrough.link(script.inputs['headpose_pass'])

#################################################################################################################
#NN mask 1
print("Creating Mask recognition NN ImageManip")
mask_rec_manip = pipeline.create(dai.node.ImageManip)
mask_rec_manip.initialConfig.setResize(224, 224)
#mask_rec_manip should be same as scipt node ouptput image size
mask_rec_manip.setWaitForConfigInput(True)
script.outputs['manip3_cfg'].link(mask_rec_manip.inputConfig)
script.outputs['manip3_img'].link(mask_rec_manip.inputImage)

# Second stange recognition NN---Mask
print("Creating recognition Neural Network for mask...")
mask_rec_nn = pipeline.create(dai.node.NeuralNetwork)
mask_rec_nn.setBlobPath(blobconverter.from_zoo(name="sbd_mask_classification_224x224", zoo_type="depthai", shaves=6))
mask_rec_manip.out.link(mask_rec_nn.input)


mask_rec_xout = pipeline.create(dai.node.XLinkOut)
mask_rec_xout.setStreamName("mask-recognition")
mask_rec_nn.out.link(mask_rec_xout.input)
########################################################################################################################################
#NN-2 age gender recognition
print("Creating Age gender recognition NN ImageManip")
age_gender_manip = pipeline.create(dai.node.ImageManip)
age_gender_manip.initialConfig.setResize(62, 62)
age_gender_manip.setWaitForConfigInput(True)
script.outputs['manip4_cfg'].link(age_gender_manip.inputConfig)
script.outputs['manip4_img'].link(age_gender_manip.inputImage)

 # Second stange recognition NN---Age gender
print("Creating recognition Neural Network for age gender...")
age_gender_nn = pipeline.create(dai.node.NeuralNetwork)
age_gender_nn.setBlobPath(blobconverter.from_zoo(name="age-gender-recognition-retail-0013", shaves=6))
age_gender_manip.out.link(age_gender_nn.input)

age_gender_xout = pipeline.create(dai.node.XLinkOut)
age_gender_xout.setStreamName("ageGender-recognition")
age_gender_nn.out.link(age_gender_xout.input)
##############################################################
#NN-3 Emotion Recognition
print("Creating emotion recognition NN ImageManip")
emotion_manip = pipeline.create(dai.node.ImageManip) #manip_manip---->emotion_manip
emotion_manip.initialConfig.setResize(64, 64)
emotion_manip.setWaitForConfigInput(True)
script.outputs['manip5_cfg'].link(emotion_manip.inputConfig)#image_manip_script--->script
script.outputs['manip5_img'].link(emotion_manip.inputImage)

# This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
# face that was detected by the face-detection NN.
print("Creating recognition Neural Network for emotion...")
emotions_nn = pipeline.create(dai.node.NeuralNetwork)
emotions_nn.setBlobPath(blobconverter.from_zoo(name="emotions-recognition-retail-0003", shaves=6))
emotion_manip.out.link(emotions_nn.input)

emotion_recognition_xout = pipeline.create(dai.node.XLinkOut)
emotion_recognition_xout.setStreamName("emotion-recognition")
emotions_nn.out.link(emotion_recognition_xout.input)

##############################################################################333


print("Creating face recognition ImageManip/NN")

face_rec_manip = pipeline.create(dai.node.ImageManip)
face_rec_manip.initialConfig.setResize(112, 112)
face_rec_manip.setWaitForConfigInput(True)

script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
script.outputs['manip2_img'].link(face_rec_manip.inputImage)

face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
face_rec_nn.setBlobPath(blobconverter.from_zoo(name="face-recognition-arcface-112x112", zoo_type="depthai", shaves=6))
face_rec_manip.out.link(face_rec_nn.input)

arc_xout = pipeline.create(dai.node.XLinkOut)
arc_xout.setStreamName('recognition')
face_rec_nn.out.link(arc_xout.input)


with dai.Device(pipeline) as device:
    facerec = FaceRecognition(databases, args.name)
    sync = TwoStageHostSeqSync()
    text = TextHelper()

    queues = {}
        # Create output queues
    for name in ["color", "detection", "recognition","mask-recognition","ageGender-recognition","emotion-recognition"]:
        queues[name] = device.getOutputQueue(name)

    id=0
    
    while True:
        for name, q in queues.items():
                        # Add all msgs (color frames, object detections and face recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            dets = msgs["detection"].detections
            frame=cv2.flip(frame,0)
            #faces-detection:[1,2,3,4,5,6,7,]
            for i, detection in enumerate(dets):
                h=frame.shape[0]
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                
                

                features = np.array(msgs["recognition"][i].getFirstLayerFp16())
                conf, name = facerec.new_recognition(features)
                size=len(os.listdir(f"{databases}/{name}"))
                #s=len(f"{databases}/{name}")
                #print("size: ",size,"s: ",s)
                
                rec = np.array(msgs["mask-recognition"][i].getFirstLayerFp16())
                #rec = mask-recognitions[i].getFirstLayerFp16()
                index = np.argmax(log_softmax(rec))
                texts = "No Mask"
                color = (0,0,255) # Red
                if index == 1:
                    texts = "Mask"
                    color = (0,255,0) 

                #rec =ageGender-recognition[i]
                age = int(float(np.squeeze(np.array( msgs["ageGender-recognition"][i].getLayerFp16('age_conv3')))) * 100)
                gender = np.squeeze(np.array( msgs["ageGender-recognition"][i].getLayerFp16('prob')))
                gender_str = "female" if gender[0] > gender[1] else "male"
################################################################################################################################################################
                #rec = msgs["emotion-recognition"] [i]

                emotion_results = np.array(msgs["emotion-recognition"][i].getFirstLayerFp16())
                emotion_name = emotions[np.argmax(emotion_results)]


                ################### FACE DATABASE

                if  name!='unknown' and  id%10==0 and size<12:
                    
                    # img=cv2.flip(img,0)
                    img2=frame
                    blank = np.zeros(img2.shape[:2], dtype='uint8')
                    rectangle = cv2.rectangle(blank.copy(),  (bbox[0]-10,h-bbox[1]+10), (bbox[2]+10,h-bbox[3]-10), 255, -1)
                    masked = cv2.bitwise_and(img2,img2,mask=rectangle)
                    cv2.imwrite(f"{databases}/{name}/{name}${id}.png",masked)
                    
            

                id=id+1
                cv2.rectangle(frame, (bbox[0]-10,h-bbox[1]+10), (bbox[2]+10,h-bbox[3]-10), (10, 245, 10), 2)
               # text.putText(frame, f"{name} {(100*conf):.0f}%", (bbox[0] + 10,h-(bbox[3] + 35)))
                
                if conf<0.75:
                    name='unknown'
                
                #text.putText(frame, f"{name} {(100*conf):.0f}% \n {texts} \n {age} {gender_str} \n {emotion_name}", (bbox[0] + 10,h-(bbox[1] + 35)))
                text.putText(frame, f"{name} {(100*conf):.0f}%  {emotion_name}", (bbox[0] + 10,h-(bbox[1] + 35)))
                # text.putText(frame, f"{gender_str} {age}", (bbox[0] + 10,20+h-(bbox[1] + 35)))
                text.putText(frame, f"{gender_str}", (bbox[0] + 10,20+h-(bbox[1] + 35)))
               

            # f=cv2.flip(frame,0)
            ###text.putText(frame, f"{name} {(100*conf):.0f}% {texts}", (bbox[0] + 10,bbox[1] + 35))

                # text.putText(frame, f"{name} {(100*conf):.0f}%  ", (bbox[0] + 10,bbox[1] + 35))
                # y = (bbox[1] + bbox[3]) // 2
                # cv2.putText(frame, texts, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 4)
                # cv2.putText(frame, texts, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

                
                # y = (bbox[1] + bbox[3]) // 2
                # cv2.putText(frame, str(age), (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                # cv2.putText(frame, str(age), (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                # cv2.putText(frame, gender_str, (bbox[0], y + 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                # cv2.putText(frame, gender_str, (bbox[0], y + 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

                #cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                #y = (bbox[1] + bbox[3]) // 2
                #cv2.putText(frame, emotion_name, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                #cv2.putText(frame, emotion_name, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)

            
            cv2.imshow("color", cv2.resize(frame, (800,800)))

        if cv2.waitKey(1) == ord('q'):
            break