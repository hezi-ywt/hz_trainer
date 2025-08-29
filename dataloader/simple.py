from torch.utils.data import Dataset
from torch.utils.data import DataLoader
 
dataset = {0: '张三', 1:'李四', 2:'王五', 3:'赵六', 4:'陈七'}

dataloader = DataLoader(dataset, batch_size=2,shuffle=True)

for i, value in enumerate(dataloader):
    print(i, value)