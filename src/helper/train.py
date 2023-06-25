import tqdm
import torch.nn.functional as F


def train(model, train_loader, optimizer, num_epoch, device, scheduler=None, accum_size= 1, start_epoch=1):
    model.train()
    model.to(device)
    total_loss = 0
    for epoch in range(start_epoch, num_epoch + 1):
      print(f'Epoch {epoch}')
      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)
          print(output)
          loss = F.cross_entropy(output,target) / accum_size
          total_loss += loss.item() * accum_size
          loss.backward(retain_graph=True)
          if batch_idx % accum_size == 0: optimizer.step()
          print(f'Batch {batch_idx} Loss {loss.item()}',end="\r")
      print('Total Loss', total_loss)
      if scheduler: scheduler.step()