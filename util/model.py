import torch, torchvision

def ResNet(model_name = "resnet50", num_classes = 2, pretrained = False):
    if model_name.lower() == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
    elif model_name.lower() == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
    elif model_name.lower() == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
    else:
        print("Model name is not verified.")
        return
    num_ftrs = model.fc.in_features
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

def Optimizer(model,initial_lr,mode = "sgd"):
    if mode.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.005)
    elif mode.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.05)
    else:
        print("Optimizer cannot be defined.")
        return
    return optimizer