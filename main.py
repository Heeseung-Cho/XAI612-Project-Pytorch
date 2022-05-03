import random
import numpy as np
import os
import argparse
import pickle
import torch, torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from util.dataloader import WrapperDataset
from util.transform import get_Transforms
from util.model import ResNet, Optimizer
from util.train import fit, test
import wandb
from sklearn.model_selection import StratifiedKFold

## Argparse
parser = argparse.ArgumentParser(description='XAI612-Project', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


## Fix setting
parser.add_argument('--epochs', default=300, type=int,
                    help='Epochs')
parser.add_argument('--train_path', default="retina1_trainvalid/trainvalid", type=str,
                     help='Set source(train) path')
parser.add_argument('--test_path', default="test_big", type=str,
                     help='Set test path')
parser.add_argument('--batch_size', default=32, type=int,
                     help='Batch Size')

## Changable settings                     
parser.add_argument('--save_path', default='baseline', type=str,
                     help='Set save path')
parser.add_argument('--image_size', default=224, type=int,
                     help='Image Size')
parser.add_argument('--model', default="resnet18", type=str,
                     help='Choose models, resnet 18,32,50')
parser.add_argument('--augmentation', default="none", type=str,
                     help='Set augmentation, none, weak, hard')
parser.add_argument('--optimizer', default="sgd", type=str,
                     help='Set optimizer, sgd, adamw')
parser.add_argument('--initial_lr', default=4e-3, type=float,
                     help='Set initial learning rate')
parser.add_argument('--pretrained', default=False, type=bool,
                     help='Set whether using pretrain')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


def main():
    global args
    args = parser.parse_args()
    project_wandb = "xai612_project"
    run_name = args.save_path    

    ## Initial params
    trainvalid_path=args.train_path
    test_path=args.test_path
    root_path = os.getcwd()
    batch_size = args.batch_size
    img_height = img_width = args.image_size
    save_path = root_path + f'/saved_model/{args.save_path}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    n_folds = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'CUDA is available. Your device is {device}.')
    else:
        print(f'CUDA is not available. Your device is {device}. It can take long time training in CPU.')    

    ## Dataloader
    data_transforms = get_Transforms(img_height, img_width, mode = args.augmentation)
    train_valid_dataset = torchvision.datasets.ImageFolder(trainvalid_path)    
    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=data_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
    
    y = np.loadtxt("retina1_trainvalid/labels_trainvalid.txt")  ## For Stratify. If not, delete this part.
    splits = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 42)

    ## K-Fold validation
    results = dict()      
    for fold_, (train_idx, valid_idx) in enumerate(splits.split(train_valid_dataset, y)):
        print("fold n°{}".format(fold_+1))        
        modelPath = os.path.join(save_path, f'bestModel_fold{fold_+1}.pt')
        run_note = run_name + f'_fold{fold_+1}'
        wandb.init(project = project_wandb, config = args, name = run_note)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(
            WrapperDataset(train_valid_dataset, transform=data_transforms['train']),batch_size = batch_size, sampler=train_sampler,num_workers = 8)
        valid_loader = DataLoader(
            WrapperDataset(train_valid_dataset, transform=data_transforms['val']),batch_size = batch_size, sampler=valid_sampler,num_workers = 4)

        ## Model
        model = ResNet(model_name=args.model, num_classes = 2, pretrained = args.pretrained)
        wandb.watch(model, log="all")
        # train/validate now
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Optimizer(model = model,initial_lr = args.initial_lr, mode = args.optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5)                
        history = fit(epochs = args.epochs,
                      model = model,
                      train_loader = train_loader,
                      val_loader = valid_loader,
                      criterion = criterion, 
                      optimizer = optimizer, 
                      path = modelPath, 
                      scheduler= scheduler,
                      earlystop = 20,                      
                      device = device)
        results[fold_+1] = history['val_acc'][-1]
        wandb.finish()

    # Print fold results
    print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {n_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value*100} %')
        sum += value
    print(f'Average: {sum/len(results.items())*100} %')


if __name__ == "__main__":
    set_seed(42)
    main()
