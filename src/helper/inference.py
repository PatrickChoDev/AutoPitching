import torch

def validate(model, data_loader, device, criterion):
    model.eval()
    model.to(device)
    total_loss = 0
    with torch.no_grad():
        for (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return loss.item()

def inference(model, data, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
      return model(data.to(device)).cpu()