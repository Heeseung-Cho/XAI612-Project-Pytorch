import os
from glob import glob
import shutil

trainvalid = "retina1_trainvalid/trainvalid"
test = "test_big"
trainvalid_dir = sorted(glob(trainvalid+'/*.jpg'))
test_dir = sorted(glob(test+'/*.*'))

y = []
with open("retina1_trainvalid/labels_trainvalid.txt","r") as f:
  labels = f.readlines()
  for line in labels:
    words = line.split()
    y.append(int(words[0]))
print(len(y))


if not os.path.exists('retina1_trainvalid/trainvalid/0'):
    os.mkdir('retina1_trainvalid/trainvalid/0')
if not os.path.exists('retina1_trainvalid/trainvalid/1'):
    os.mkdir('retina1_trainvalid/trainvalid/1')

for i in range(len(y)):
  if y[i] == 0:
    shutil.move(trainvalid_dir[i],'retina1_trainvalid/trainvalid/0')
  elif y[i] == 1:
    shutil.move(trainvalid_dir[i],'retina1_trainvalid/trainvalid/1')
  else:
    pass