import random
import numpy as np
import os
import argparse
import pandas as pd
import torch, torchvision
from util.transform import get_Transforms
from util.model import ResNet
from util.train import test


## Argparse
parser = argparse.ArgumentParser(description='XAI612-Project', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


## Fix setting
parser.add_argument('--test_path', default="test_big", type=str,
                     help='Set test path')

## Changable settings                     
parser.add_argument('--save_path', default='Baseline_resnet18', type=str,
                     help='Set saved model path')
parser.add_argument('--image_size', default=224, type=int,
                     help='Image Size')
parser.add_argument('--submission_name', default="submission", type=str,
                     help='Name of submission file')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


def main():
    global args
    args = parser.parse_args()

    ## Initial params
    test_path=args.test_path
    root_path = os.getcwd()
    img_height = img_width = args.image_size
    save_path = args.save_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'CUDA is available. Your device is {device}.')
    else:
        print(f'CUDA is not available. Your device is {device}. It can take long time training in CPU.')    

    ## Dataloader
    data_transforms = get_Transforms(img_height, img_width, mode = 'none')
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(test_path, transform=data_transforms['val']), 
                                              batch_size = 1, shuffle = False)#, num_workers = 4)

    prediction = dict()
    for i in range(5):
        model_path = root_path + f'/saved_model/{save_path}/bestModel_fold{i+1}.pt'
        model = ResNet(model_name="resnet18", num_classes = 2, pretrained = False)
        model.load_state_dict(torch.load(model_path))
        pred = test(model,test_loader, device)                
        prediction[i+1] = pred
    
    prediction = np.mean(np.array(list(prediction.values())),axis = 0)
    prediction = [1 if p>=0.5 else 0 for p in prediction]
    submission = pd.read_csv('submission.csv')
    submission['predicted_label'] = prediction
    submission.to_csv(root_path + f'/saved_model/{save_path}/{args.submission_name}.csv',index = False)

if __name__ == "__main__":
    set_seed(42)
    main()
