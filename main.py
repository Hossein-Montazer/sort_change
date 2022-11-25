# used this link :https://medium.com/@jarrodmccarthy12/object-tracking-with-yolov5-and-sort-589e3767f85c
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from sort import *
import cv2
from pathlib import Path


# if __name__=='__main__':

model = torch.hub.load('.', 'custom', './yolov5s.pt', source='local')

# model=torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
# model.float()
# model.eval()

# !wget "https://drive.google.com/uc?export=download&id=1In4inopruHy8WbH3XZMb660T3QDzzs7z" -O traffic.mp4
vid=cv2.VideoCapture('traffic.mp4')
mot_tracker = Sort()
save_path='rnn2'
ret,img=vid.read()
fps, w, h = 30, img.shape[1], img.shape[0]
save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
i=0
while(True):
  ret,img=vid.read()
  if ret!=1:
    break
  preds = model(img)
  detections=preds.pred[0].cpu().numpy()
  track_id=mot_tracker.update(detections)
  for j in range(len(track_id.tolist())):
    coords = track_id.tolist()[j]
    point = detections.tolist()[j]

    cv2.rectangle(img, (int(point[0]),int(point[1])), (int(point[2]),int(point[3])), (0,0,200), 1, cv2.LINE_AA)  # filled
    x1,y1,x2,y2 = int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])
    name_id=int(coords[4])
    name='id:{}'.format(str(name_id))
    cv2.rectangle(img,(x1,y1),(x2,y2),name_id,2)
    cv2.putText(img,name,(x1,y1-1),cv2.FONT_HERSHEY_SIMPLEX,.9,name_id,2)

  vid_writer.write(img)
  i+=1

  if cv2.waitKey(1)&0xFF==ord('q'):
    break
vid_writer.release()
vid.release()
cv2.destroyAllWindows()