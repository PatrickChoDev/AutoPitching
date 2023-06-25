from data.dataset import SoundDS
import torch
from model.head import RawHead


sd= SoundDS('')
sd.add('/home/patrick/Workspace/Projects/SHT7/AutoPitching/data/evil', 1)
sd.add('/home/patrick/Workspace/Projects/SHT7/AutoPitching/data/heaven', 0)


train_set, valid_set = torch.utils.data.random_split(sd, [int(len(sd)*0.8), len(sd)-int(len(sd)*0.8)])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=8, shuffle=False)



model =  RawHead(80,8,1)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.1)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

model.train()
model.to(device)
accum_size = 1
start_epoch = 1
num_epoch = 10

for epoch in range(start_epoch, num_epoch + 1):
  print(f'Epoch {epoch}')
  total_loss = 0
  for data, target in train_loader:
      data, target = data.to(device), target.to(device)
      optim.zero_grad()
      outpu,conf = model(data)
      loss = torch.nn.functional.cross_entropy(conf.sigmoid(),target)
      total_loss += loss.item() * accum_size
      loss.backward()
      optim.step()
      print(f'Loss {loss.item()}',end="\r")
  print('Total Loss', total_loss/len(train_loader))
  if scheduler: scheduler.step()

torch.save(model,'models/model-100.pth')