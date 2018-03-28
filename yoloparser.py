import numpy as np
import math

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

def logistic_activate(x):
    """Logistic sigmoid activation function."""
    return 1.0/(1.0 + math.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class yolov2parser():
    """
    Parsing 1D tensor output into detected objects by Yolo v2.

    Parameters
    ----------
    output_blob : tensor output to be parse
    dim : image dimension (width or height, since the image is rectangle)
    nclass : number of classes 
    nbox : number of anchor boxes
    classes_name : list of classes name in string format
    biases : list of anchor boxes parameters

    Functions
    ---------
    __check_input: checking input sie
    __reorder: reorder the output
    interpret: convert output tensor to bounding boxes with specified confidence threshold & NMS parameters

    """
    def __init__(self,output_blob,dim,nclass,nbox,classes_name,biases):
        self.output_blob = output_blob
        self.dim = dim
        self.nclass = nclass
        self.nbox = nbox
        self.classes_name = classes_name
        self.biases = biases
        self.pred_size = 5 + self.nclass # prediction size (x,y,w,h,conf score + class probs)
        self.grid_size = self.dim * self.dim
        self.total_pred_size = self.pred_size*self.nbox # length of total N bounding boxes: size of predictions * number of anchor box 
        self.total_objects = self.grid_size * self.nbox #number of predictions made, each grid will predict N bounding boxes

    def __check_input(self):
        if self.output_blob.size != (self.grid_size*self.nbox*(self.pred_size)):
            raise Exception('Output size mismatch!')

    def _reorder(self):
        """
        Reorder the output so as to order the elements that previously apart by grid size
        """
        self.__check_input()
        new_blob =[]
        for i in range(self.grid_size):
            p = i
            #print ("index: {}").format(i)
            for _ in range (self.total_pred_size):
                new_blob.append(self.output_blob[p])
                #print ("p: {}").format(p)
                p += self.grid_size
        self.output_blob = new_blob

    def interpret(self,threshold,nms,image_width,image_height,reorder=True):
        """
        Interpret the output blob, do non-max-suppresion and output all detected objects 

        Parameters
        ----------
        threshold : class probabilities threshold
        nms : IoU threshold for 2 boxes to be supressed
        image_width : width of the input image
        image_height : height of the input image
        reorder : flag to determine whether to reorder the output or not

        Returns
        -------
        final_result : list of detected objects 
        """
        final_result = [] #list for all detected objects that have been suppressed 
        if reorder:
            self._reorder()
        # get all boxes from the grids 
        detected_dict = {}
        for class_num in range (self.nclass): #initialise the dictionary with key all of class, and value a list of detected object (which is now empty)
            detected_dict[self.classes_name[class_num]] = []
        
        for i in range(self.total_objects): # number of predictions 12*12*5, total_objects
            index = i * self.pred_size

            # box params
            n = i % self.nbox #index for box 
            row = (i/self.nbox) / self.dim
            col = (i/self.nbox) % self.dim
            #print ("index: {} row: {} col: {} box_num: {}").format(index,row,col,n)
            x = (col + logistic_activate(self.output_blob[index+0])) / self.dim #ditambah col sama row terus dibagi blockwd(dim) supaya relatif terhadap grid itu, range nya jadi 0-1
            y = (row + logistic_activate(self.output_blob[index+1])) / self.dim
            w = math.exp(self.output_blob[index+2]) * self.biases[2*n] / self.dim
            h = math.exp(self.output_blob[index+3]) * self.biases[2*n+1] / self.dim
            box = Boxes(x,y,w,h)
            #print(str(Boxes.box_list[i]))

            # scale (confidence score)
            scale = logistic_activate(self.output_blob[index + 4])
            #scales.append(scale)

            # class probabilities
            class_probs_start = index + 5
            class_probs_end = class_probs_start + self.nclass
            class_probs = self.output_blob[class_probs_start:class_probs_end]
            #print ("before softmax:{}").format(class_probs)
            class_probs = softmax(class_probs)
            #print ("after softmax:{}").format(class_probs) # softmax function ok
            scaled_class_probs = [prob * scale for prob in class_probs]

            # save only box that has class probs > threshold to a dictionary of respective class detections
            for j in range (self.nclass):
                if scaled_class_probs[j] > threshold:
                    #print ("row:{} col:{} box_num:{} looking for class:{} confidence:{}").format(row,col,n,j,scaled_class_probs[j])
                    new_list = detected_dict.get(self.classes_name[j])
                    #print (new_list)
                    new_list.append(DetectedObject(box,scaled_class_probs[j],self.classes_name[j],image_width,image_height)) # detected object already in form of xmin.xmax,ymin,ymax relative to image size
                    #print (new_list)
                    detected_dict[self.classes_name[j]] = new_list

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
        return final_result
