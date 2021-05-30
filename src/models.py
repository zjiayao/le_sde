import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleConv(nn.Module):
    def __init__(self, input_shape, n_classes, has_pre_softmax_ly=False, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.has_pre_softmax_ly = has_pre_softmax_ly
        if has_pre_softmax_ly:
            self.pre_softmax_ly = nn.Linear(n_classes,n_classes,bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.has_pre_softmax_ly:
            x = self.pre_softmax_ly(x)
        return x
    
class SimpleConv3d(nn.Module):
    def __init__(self, input_shape, n_classes, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def gen_model(data_shape, n_classes, lr=0.005, model_func=SimpleConv, rand_H=False):
    model = model_func(input_shape=data_shape, n_classes=n_classes, has_pre_softmax_ly=rand_H).double()
    if rand_H:
        model.pre_softmax_ly.requires_grad = False
        model.pre_softmax_ly.weight.data = torch.tensor(get_ortho_mat(n_classes),dtype=torch.double)
    loss_func = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0)
    return model, loss_func, opt
