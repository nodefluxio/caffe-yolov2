import caffe
GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
#caffe.set_mode_gpu()
#caffe.set_device(GPU_ID)
caffe.set_mode_cpu()
from datetime import datetime
import numpy as np
import sys, getopt
import cv2
import math

default_yolo_header = "comp4_det_test_"
biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]

class Boxes():
    box_list=[]
    def __init__(self,x,y,w,h):
        Boxes.box_list.append(self)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    #this is just for output formatting
    def __str__(self):
        return "x,y: {},{} width: {} height: {}".format(self.x,self.y,self.w,self.h)

class DetectedObject():
    def __init__(self,box,conf,object_class,imgw,imgh):
        self.xmin = int((box.x - (box.w/2.0))*imgw)
        self.xmax = int((box.x + (box.w/2.0))*imgw)
        self.ymin = int((box.y - (box.h/2.0))*imgh)
        self.ymax = int((box.y + (box.h/2.0))*imgh)
        self.conf = conf
        self.object_class = object_class
    def __str__(self):
        return "(xmin,xmax,ymin,ymax): ({},{},{},{}) conf: {} object_class:{}".format(self.xmin,self.xmax,self.ymin,self.ymax,self.conf,self.object_class)

def reorderOutput(input_arr):
    #k: size of predictions * number of anchor box --> 25 x 5
    #size of predictions = 5 + number of classes --> 25
    #n: grid size * grid size --> 12 x 12
    pred_size =  25
    grids = 12 * 12
    boxes = 5
    total_pred_size = pred_size*boxes #125

    new_arr =[]
    for i in range(grids):
        p = i
        #print ("index: {}").format(i)
        for j in range (total_pred_size):
            new_arr.append(input_arr[p])
            #print ("p: {}").format(p)
            p += grids
    return new_arr

def logisticActivate(x):
    return 1.0/(1.0 + math.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def drawRectangles(img, box,color): # function to draw rectangles
    #color is a tuple of rgb value
    #print (img.shape)
    if (color =="green"):
        #box = convertYoloBox(bbox,img.shape[0],img.shape[1])
        color = (0,255,0)
    elif (color == "red"):
        #box = convertIEBox(bbox,img.shape[0],img.shape[1])
        color = (0,0,255)
    else:
        color = (255,0,0)
    #print (box)
    # rectangle (image,(x1,y1),(x2,y2),(color index),(boldness)) top-left corner and bottom-right corner
    img = cv2.rectangle(img,(box[0],box[3]),(box[1],box[2]),color,2)
    #cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("window",img)
    #cv2.waitKey(5000)

def calculate_iou(boxA, boxB):

    # determine the (x, y) coordinates of the intersection rectangle
    xA = max(boxA.xmin,boxB.xmin)
    xB = min(boxA.xmax,boxB.xmax)
    yA = max(boxA.ymin,boxB.ymin)
    yB = min(boxA.ymax,boxB.ymax)
    #print ("xA xB yA yB {} {} {} {}").format(xA,xB,yA,yB)
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1)*(yB - yA + 1)
    #print ("interArea {}").format(interArea)
    # compute the area of union
    boxAArea = (boxA.xmax-boxA.xmin+1) * (boxA.ymax-boxA.ymin+1)
    boxBArea = (boxB.xmax-boxB.xmin+1) * (boxB.ymax-boxB.ymin+1)

    unionArea = float(boxAArea+boxBArea-interArea)
    iou = interArea / unionArea
    return iou
 
def interpretOutputV2(output,image_width, image_height):
    classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    threshold = 0.2
    nms = 0.4
    iou_threshold = 0.5
    classes = 20
    num_box = 5
    out_w = 12
    out_h = 12
    blockwd = 12
    #scales = [] #confidence score that there is an object on that grid cell
    
    pred_size = 5 + classes #size of predictions = 5 + number of classes --> 25
    grid_size = out_w*out_h #n: grid size * grid size --> 12 x 12
    total_pred_size = pred_size * num_box #length of total N bounding boxes: size of predictions * number of anchor box --> 25 x 5
    total_objects = out_w * out_h * num_box #number of predictions made, each grid will predict N bounding boxes.
    final_result = [] #list for all detected objects that have been suppressed 

    # get all boxes from the grids 
    detected_dict = {}
    for class_num in range (classes): #initialise the dictionary with key all of class, and value a list of detected object (which is now empty)
        detected_dict[classes_name[class_num]] = []
    
    for i in range(total_objects): # number of predictions 12*12*5, total_objects
        index = i * pred_size

        # box params
        n = i % num_box #index for box 
        row = (i/num_box) / out_w
        col = (i/num_box) % out_w
        #print ("index: {} row: {} col: {} box_num: {}").format(index,row,col,n)
        x = (col + logisticActivate(output[index+0])) / blockwd #ditambah col sama row terus dibagi blockwd supaya relatif terhadap grid itu, range nya jadi 0-1
        y = (row + logisticActivate(output[index+1])) / blockwd
        w = math.exp(output[index+2]) * biases[2*n] / blockwd
        h = math.exp(output[index+3]) * biases[2*n+1] / blockwd
        box = Boxes(x,y,w,h)
        #print(str(Boxes.box_list[i]))

        # scale (confidence score)
        scale = logisticActivate(output[index + 4])
        #scales.append(scale)

	    # class probabilities
        class_probs_start = index + 5
        class_probs_end = class_probs_start + classes
        class_probs = output[class_probs_start:class_probs_end]
        #print ("before softmax:{}").format(class_probs)
        class_probs = softmax(class_probs)
        #print ("after softmax:{}").format(class_probs) # softmax function ok
        scaled_class_probs = [prob * scale for prob in class_probs]
        #print ("scale:{}").format(scale)
        #print ("scaled class probabilities:{}").format(class_probs)
        # detected_list.append(DetectedObject(box,scale,scaled_class_probs))
        #for detected in detected_list:
        #    print (str(detected))

        # save only box that has class probs > threshold to a dictionary of respective class detections
        for j in range (classes):
            if scaled_class_probs[j] > threshold:
                #print ("row:{} col:{} box_num:{} looking for class:{} confidence:{}").format(row,col,n,j,scaled_class_probs[j])
                new_list = detected_dict.get(classes_name[j])
                #print (new_list)
                new_list.append(DetectedObject(box,scaled_class_probs[j],classes_name[j],image_width,image_height)) # detected object already in form of xmin.xmax,ymin,ymax relative to image size
                #print (new_list)
                detected_dict[classes_name[j]] = new_list

    for key,value in detected_dict.items(): #print all the dict value, dict value is a list of object instances
        if (value): # if there are boxes for this object
            prev_max_conf = 0.0
            max_box = None
            for box in value:
                #print(prev_max_conf)
                #print ("class type: {} boxes: {}").format(key,str(box))
                #find the highest confidence score box in the list of boxes for a class
                if box.conf >= prev_max_conf:
                    max_box = box
                    prev_max_conf = box.conf
            #print (max_box)
            final_result.append(max_box)

        #iterate over the other boxes, filtering out overlapped boxes (NMS)
            for box in value:
                iou = calculate_iou(max_box,box)
                #print (iou)
                if(iou < nms):
                    final_result.append(box)

    # for result in final_result:
    #     print (result)
    return final_result

def show_results(img,results, img_width, img_height, image_id):
    img_cp = img.copy()
    disp_console = True
    imshow = False
    for i in range(len(results)):
        conf = results[i].conf
        xmin = results[i].xmin
        xmax = results[i].xmax
        ymin = results[i].ymin
        ymax = results[i].ymax
        if xmin<0.0:
            xmin = 0.0
        if ymin<0.0:
            ymin = 0.0
        if xmax>img_width:
		    xmax = img_width
        if ymax>img_height:
		    ymax = img_height
        if  imshow:
			cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(0,255,0),2)
		#	print xmin, ymin, xmax, ymax
			cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
			cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)	
        fo = open("/Users/richardharmadi/nodeflux/validation_app/caffe_voc_validate/results/"+default_yolo_header+results[i].object_class+".txt","a")
        fo.write("{} {:f} {:f} {:f} {:f} {:f}\n".format(image_id,conf,xmin,ymin,xmax,ymax))
        #print ("{} {:f} {:f} {:f} {:f} {:f}\n".format(image_id,conf,xmin,ymin,xmax,ymax))
	#if imshow :
	#	cv2.imshow('YOLO detection',img_cp)
	#	cv2.waitKey(1000)

def main(argv):
    model_filename = ''
    weight_filename = ''
    img_filename = ''
    try:
	opts, args = getopt.getopt(argv, "hm:w:i:")
	print opts
    except getopt.GetoptError:
	print 'yolo_main.py -m <model_file> -w <output_file> -i <img_file>'
	sys.exit(2)
    for opt, arg in opts:
	if opt == '-h':
	    print 'yolo_main.py -m <model_file> -w <weight_file> -i <img_file>'
            sys.exit()
	elif opt == "-m":
	    model_filename = arg
	elif opt == "-w":
	    weight_filename = arg
	elif opt == "-i":
	    img_filename = arg
    print 'model file is "', model_filename
    print 'weight file is "', weight_filename
    print 'image file is "', img_filename
    image_name = img_filename.split("/")[-1]
    image_id = image_name.split(".")[0]
    net = caffe.Net(model_filename, weight_filename, caffe.TEST)
    img = caffe.io.load_image(img_filename) # load the image using caffe io
    inputs = img
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    start = datetime.now()
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
    end = datetime.now()
    elapsedTime = end-start
    print 'total time is " milliseconds', elapsedTime.total_seconds()*1000
    #for k,v in out.iteritems():
    #    print (k)
	#print(out['conv9'].shape)
	#print(out['conv9'][0].shape)
    out1d = reorderOutput(out['conv9'][0].flatten())
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = interpretOutputV2(out1d,img.shape[1],img.shape[0])
    show_results(img_cv,results, img.shape[1], img.shape[0],image_id)
    #cv2.waitKey(10000)

if __name__=='__main__':	
    main(sys.argv[1:])
