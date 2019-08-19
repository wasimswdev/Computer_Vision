import sys
import cv2
from random import randint

        

trackerType = cv2.TrackerCSRT_create()
      

  # Set video to load
videoPath = "airport.mp4"
  
  # Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)
 
  # Read first frame
success, frame = cap.read()
  # quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

  ## Select boxes
bboxes = []
colors = [] 

  # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
  # So we will call this function in a loop till we are done selecting all objects
while True:

    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
  bbox = cv2.selectROI('MultiTracker', frame)
  bboxes.append(bbox)
  colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
  print("Press q to quit selecting boxes and start tracking")
  print("Press any other key to select next object")
  k = cv2.waitKey(0) & 0xFF
  if (k == 113):  # q is pressed
    break
  
print('Selected bounding boxes {}'.format(bboxes))

  ## Initialize MultiTracker
  # There are two ways you can initialize multitracker
  # 1. tracker = cv2.MultiTracker("CSRT")
  # All the trackers added to this multitracker
  # will use CSRT algorithm as default
  # 2. tracker = cv2.MultiTracker()
  # No default algorithm specified

  # Initialize MultiTracker with tracking algo
  # Specify tracker type
  
  # Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

  # Initialize MultiTracker 
for bbox in bboxes:
  multiTracker.add(trackerType, frame, bbox)


  # Process video and track objects
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break
    
    # get updated location of objects in subsequent frames
  success, boxes = multiTracker.update(frame)

    # draw tracked objects
  for i in range(len(boxes)):
    new_x = abs(boxes[i][0])    
    new_y = abs(boxes[i][1])
    new_w = abs(boxes[i][2]) 
    new_h = abs(boxes[i][3]) 

    old_x = bboxes[i][0]
    old_y = bboxes[i][1]
    old_w = bboxes[i][2]
    old_h = bboxes[i][3]
    
  if abs(old_x - new_x) >= 120:
    print('object moved')

  elif abs(old_y - new_y) >= 120:
    print('object moved')

  elif abs(old_w - new_w) >= 120:
    print('object moved')

  elif abs(old_h - new_h) >= 120:
    print('object moved')

  p1 = (int(boxes[i][0]), int(boxes[i][1]))
  p2 = (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3]))
  cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
  cv2.imshow('MultiTracker', frame)
    

    # quit on ESC button
  if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    break

if not success:
  print("Object did not moved")