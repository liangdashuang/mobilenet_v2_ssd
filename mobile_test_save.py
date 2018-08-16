import numpy as np  
import sys,os  
import cv2
import pdb
caffe_root = '/home/will/deepLearning/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


net_file= '../../my_MobileNetSSDV2_deploy.prototxt'  
caffe_model='../../snapshot/mobilenet_iter_300000.caffemodel'
test_dir = "/sata1/liangdas_ssd/2_50mm_split"
result_dir = "../result_JPG/"
if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_gpu()
caffe.set_device(2)
net = caffe.Net(net_file,caffe_model,caffe.TEST)  
CLASSES = ('background','aeroplane', 'bicycle')
def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    for b in box:
        for i in range(4):
            if b[i] < 0: 
                b[i]=0
        if b[0] > w:
           b[0] = w
        if b[2] > w:
           b[2] = w
        if b[1] > h:
           b[1] = h
        if b[3] > h:
           b[3] = h
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       if conf[i] < 0.3:
           continue
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0),5)
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])

       if int(cls[i]) == 1:
           cv2.rectangle(origimg, p1, p2, (0,0,255),5)
           cv2.putText(origimg, title, p3, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
       else:
           cv2.rectangle(origimg, p1, p2, (0,255,255),5)
           cv2.putText(origimg, title, p3, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

       #pdb.set_trace()
       cv2.imwrite(result_dir+imgfile.split('/')[-1],origimg)
       print result_dir+imgfile.split('/')[-1]
#cv2.imshow("SSD", origimg)
 
#k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
#if k == 27 : return False
#return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
