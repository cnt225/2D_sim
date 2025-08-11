import torch
from torch import nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # k를 포인트 수에 맞게 조정
    num_points = x.size(2)
    k_actual = min(k, num_points)
 
    idx = pairwise_distance.topk(k=k_actual, dim=-1)[1]   # (batch_size, num_points, k_actual)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    # k를 포인트 수에 맞게 조정
    k_actual = min(k, num_points)
    
    if idx is None:
        idx = knn(x, k=k_actual)   # (batch_size, num_points, k_actual)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k_actual, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k_actual, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class DGCNN(nn.Module):
    def __init__(self, kwargs, output_channels=40):
        super(DGCNN, self).__init__()
        self.kwargs = kwargs
        self.k = kwargs["k"]
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(kwargs["emb_dims"])

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, kwargs["emb_dims"], kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        if 'mode' in self.kwargs:
            mode = self.kwargs.get('mode')
            if mode == 'onlymax':
                x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            elif mode == 'onlyavg':
                x = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            else:
                x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
                x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
                x = torch.cat((x1, x2), 1)
        else:
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            x = torch.cat((x1, x2), 1)
        
        return x