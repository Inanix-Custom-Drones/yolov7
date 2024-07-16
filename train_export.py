from roboflow import Roboflow
import subprocess
import glob
from IPython.display import Image, display
import torch

rf = Roboflow(api_key="tdekoGhlwhnZPAxPzvb6")

#CoHoMa cubes rouges
#rfVersion = 7
#rfWorkspace = "stage-hsrz1"
#rfName = "redbox-2f069"
#rfFolder = "redbox-7"

#Dronathlon bateaux vus du ciel
rfVersion = 3
rfWorkspace = "cv-stuff"
rfName = "aerial-ship-detection"
rfFolder = "Aerial-Ship-Detection-3"

project = rf.workspace(rfWorkspace).project(rfName)
version = project.version(rfVersion)

dataset = version.download("yolov7")

subprocess.run("python train.py --batch 8 --epochs 55 --data " + rfFolder + "/data.yaml --weights 'yolov7_training.pt' --device 0", shell=True)
subprocess.run("python detect.py --weights runs/train/exp/weights/best.pt --conf 0.1 --source " + rfFolder + "/test/images", shell=True)

i = 0
limit = 10000 # max images to print
for imageName in glob.glob('./runs/detect/exp/*.jpg'): #assuming JPG
    if i < limit:
      display(Image(filename=imageName))
      print("\n")
    i = i + 1   

print("--------------------------------")
print("CONVERTING TO TORCHSCRIPT")
print("--------------------------------")

subprocess.run("python export.py --weights ./runs/train/exp/weights/best.pt", shell=True)

subprocess.run("zip -r export.zip runs/detect", shell=True)
subprocess.run("zip -r export.zip runs/train/exp/weights/best.pt", shell=True)
subprocess.run("zip export.zip runs/train/exp/*", shell=True)
