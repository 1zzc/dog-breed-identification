
import torch

import os
from d2l import torch as d2l
import matplotlib.pylab as plt


labelPath = './labels.csv'

labels_dict =  d2l.read_csv_labels(labelPath)
print(len(labels_dict))


ans = list(set(list(labels_dict.values())))
ans.sort()
label2num = {ans[i]: i for i in range(len(ans)) }

# print(label2num)

# assert False
import torchvision
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms

class DOGDataset(Dataset):
  def __init__(self,imgpath,labels_dict,label2num):
    self.imgpath = imgpath
    self.imgNames = os.listdir(imgpath)
    self.labels_dict = labels_dict
    self.label2num = label2num
    self.transforms = torchvision.transforms.Compose([
      torchvision.transforms.RandomResizedCrop(224),
      torchvision.transforms.RandomHorizontalFlip(),
      # 随机改变亮度、对比度 饱和度
      torchvision.transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])
    ])

  def __len__(self):
    return len(self.imgNames)

  def __getitem__(self,idx):
    img = Image.open(os.path.join(self.imgpath,self.imgNames[idx]))

    img = self.transforms(img)

    imgName = self.imgNames[idx][:-4]

    label = self.labels_dict[imgName]

    return img,self.label2num[label]

dogDataset = DOGDataset('./train',labels_dict,label2num)



train_size = int(len(dogDataset) * 0.9)
validate_size = len(dogDataset)-train_size



train_dataset, validate_dataset = torch.utils.data.random_split(dogDataset, [train_size, validate_size])

# print(len(train_dataset),'hjhhh',len(validate_dataset))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,drop_last=False)
validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=True,num_workers=4,drop_last=True)


devices =  d2l.try_all_gpus()


def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet101(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

net = get_net(devices)

trainer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad),lr=1e-4,momentum=0.9,weight_decay=1e-4)
# net = nn.DataParallel(net,devices_ids=devices[0]).to(devices[0])


loss = nn.CrossEntropyLoss(reduction='none')


scheduler = torch.optim.lr_scheduler.StepLR(trainer, 2, 0.9)
num_batches, timer = len(train_loader), d2l.Timer()

legend = ['train loss']
num_epochs = 100
# d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=legend)

def evaluate_loss(data_iter,net,device):
  l_sum,n =0.0,0
  for features,labels in data_iter:
    features,labels = features.to(device),labels.to(device)
    out = net(features)
    l_sum +=loss(out,labels).sum()
    n+=labels.numel()
  return (l_sum/n).to('cpu')




# for epoch in range(num_epochs):
#   metric = d2l.Accumulator(2)
#   for i,(features,labels) in enumerate(train_loader):
#     features,labels = features.to(devices[0]),labels.to(devices[0])
#     # 梯度清零
#     trainer.zero_grad()
#     out = net(features)
#     l = loss(out,labels).sum()
#     l.backward()
#     trainer.step()
#     train_acc = (out.argmax(dim=1)==labels).sum().cpu().item()
#     print('epoch:{}/i:{}train loss{} train_acc{}'.format(epoch,i,l,train_acc))
#   valid_loss =  evaluate_loss(validate_loader,net,devices[0])
#   print('epoch:{} valid loss{}'.format(epoch,valid_loss))

# torch.save(net,'gou.pth')










class TestDataset(Dataset):
  def __init__(self,imgpath):
    self.imgpath = imgpath
    self.imgNames = os.listdir(imgpath)
    self.transforms = torchvision.transforms.Compose([
torchvision.transforms.Resize(256),
# 从图像中⼼裁切224x224⼤⼩的图⽚
torchvision.transforms.CenterCrop(224),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])
    ])



  def __len__(self):
    return len(self.imgNames)

  def __getitem__(self,idx):
    img = Image.open(os.path.join(self.imgpath,self.imgNames[idx]))
    # img=img.resize((256,256),Image.ANTIALIAS) #统一图片尺寸
    # trans = transforms.ToTensor() #转换为tensor类型
    img = self.transforms(img)

    imgName = self.imgNames[idx][:-4]

    return img,imgName

dogtest = TestDataset('./test')
testLoader=DataLoader(dogtest, batch_size=64, shuffle=False, num_workers=4,drop_last=False)

pred = []
net = torch.load('gou.pth')
idx = []


for img,imgName in testLoader:
  out = nn.functional.softmax(net(img.to(devices[0])),dim=1)
  pred.extend(out.cpu().detach().numpy())
  idx.extend(list(imgName))
  # print(len(pred))
 

with open('submission.csv','w') as f:
  f.write('id,'+','.join(ans)+'\n')
  for i,out in zip(idx,pred):
    f.write(i + ',' + ','.join(
            [str(num) for num in out]) + '\n')
    print('xie',i)








