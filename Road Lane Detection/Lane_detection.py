import argparse
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


#Getting the area where we want to Detect Lane Line
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv.fillPoly(mask, vertices, 255)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def make_line_points(y1, y2, line):
    if line is None:
        return line
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


#Draw Line on a Line Image with same Size as original Frame
def draw_line(line_img,lines):
    if lines is None:
        return line_img
    if len(lines)==0:
        return line_img
    slope_threshold=0.5
    slopes=[]
    new_lines=[]
    for line in lines:
        x1,y1,x2,y2=line[0]
        if x2-x1==0:
            slope=99 #infinite
        else:
            slope=(y2-y1)/(x2-x1)
        
        if abs(slope)>slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
        
    lines=new_lines

    #For Detecting Left and Right Side Lane (Single Average Line)
    right_line=[]
    right_weight=[]
    left_line=[]
    left_weight=[]
    for i,line in enumerate(lines):
        x1,y1,x2,y2=line[0]
        img_x_center=line_img.shape[1]/2
        intercept=y1-slopes[i]*x1
        length=np.sqrt((y2-y1)**2+(x2-x1)**2)
        if slopes[i]<0 and x1<img_x_center and x2<img_x_center:
            left_line.append((slopes[i],intercept))
            left_weight.append(length)
        elif slopes[i]>0 and x1>img_x_center and x2>img_x_center:
            right_line.append((slopes[i],intercept))
            right_weight.append(length)
    left_lane = np.dot(left_weight, left_line) / np.sum(left_weight) if left_weight else None
    right_lane = np.dot(right_weight, right_line) / np.sum(right_weight) if right_weight else None
    
    height=line_img.shape[0]
    y1 = height
    y2 = int(height * 0.75)

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    for line in [left_line,right_line]:
        if line:
            x1,y1,x2,y2=line[0]
            cv.line(line_img,(x1,y1),(x2,y2),(0,255,0),5)
    return line_img
    
#Detecct Lane for Frame Or for Images
def Detect_lane_frame(img):

    #Convert image in BGR to Gray
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #Perform blur operation to clear noise in our frame
    smoothed_image=cv.GaussianBlur(gray,(5,5),0)

    #ret,thresh=cv.threshold(smoothed_image,200,255,cv.THRESH_BINARY)

    #Detect edges where color intensity changes
    edges=cv.Canny(smoothed_image,50,150)


    height, width = edges.shape
    roi_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height),
    ]
    cropped_edges=region_of_interest(edges,np.array([roi_vertices], np.int32))

    #Find all lines associated with image to find the lanes
    lines=cv.HoughLinesP(cropped_edges,rho=1.5, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=5)
    line_img=np.zeros_like(img)
    line_img=draw_line(line_img,lines)
    combine_image=cv.addWeighted(img,0.8,line_img,1,0.0)
    
    return combine_image

#Detect lane for video or live cam and store it in output file
def Detect_Lane(cap,output):
    fourcc=cv.VideoWriter_fourcc(*'XVID')
    out=cv.VideoWriter(output,fourcc,20.0,(640,640))
    while cap.isOpened():
        ret,frame=cap.read()
        if ret:
            Detected_frame=Detect_lane_frame(frame)
            out.write(Detected_frame)
            cv.imshow('frame',Detected_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    out.release()

#Detect lane in a given single images file
def Detect_image(input,output):
    Detected_image=Detect_lane_frame(cv.imread(input))
    cv.imshow("Detecteed_image",Detected_image)
    cv.waitKey(0)
    cv.imwrite(output,Detected_image)

#Detect lane in a given video file
def Detect_video(input,output):
    cap=cv.VideoCapture(input)
    Detect_Lane(cap,output)
    cap.release()

#Detect lane using live camera input
def Detect_live(camera_parameter,output):
    cap=cv.VideoCapture(camera_parameter)
    Detect_Lane(cap,output)
    cap.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--image',help="Input Image File Location or name")
    parser.add_argument('-v','--video',help="Enter Video File Location or Name")
    parser.add_argument('-o','--output',help="Enter Name or Location of output File",default='output')
    parser.add_argument('-c','--camera',help="Enter Camera Parameter",default=0)

    args=parser.parse_args()
    video=args.video
    output=args.output
    image=args.image
    camera_parameter=args.camera

    if image:
        Detect_image(image,output+'.jpg')
    elif video:
        Detect_video(video,output+'.avi')
    else:
        Detect_live(camera_parameter,output+'.avi')
    cv.destroyAllWindows()