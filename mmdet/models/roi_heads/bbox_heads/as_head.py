import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy

@HEADS.register_module()
class asHead(nn.Module):
    def __init__(self,loss_count_num = dict(type='MSELoss',
                                            reduction='mean', loss_weight=1.0)):
        super(asHead, self).__init__()
        self.count_num_1 = nn.Linear(217600, 256)
        self.count_num_2 = nn.Linear(256, 1)
        self.trans1 = nn.Conv2d(256,256,kernel_size=7,padding=3,stride=8)
        self.trans2 = nn.Conv2d(256,256,kernel_size=5,padding=2,stride=4)
        self.trans3 = nn.Conv2d(256,256,kernel_size=3,padding=1,stride=2)
        self.trans4 = nn.Conv2d(256,256,kernel_size=3,padding=1,stride=1)
        self.count_loss = build_loss(loss_count_num)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
    def forward(self,x):
        """
        X is mutial features
        """
        mutlti_feature = x
        x1 = self.trans1(mutlti_feature[0]).view(mutlti_feature[0].size(0),-1)
        x2 = self.trans2(mutlti_feature[1]).view(mutlti_feature[1].size(0), -1)
        x3 = self.trans3(mutlti_feature[2]).view(mutlti_feature[2].size(0), -1)
        x4 = self.trans4(mutlti_feature[3]).view(mutlti_feature[3].size(0), -1)

        fc_fusion_feature = x1+x2+x3+x4
        out = self.relu(self.count_num_2(self.relu(self.count_num_1(fc_fusion_feature))))
        out = out.floor()+1
        return out

    def init_weights(self,):
        for i in [self.count_num_1,self.count_num_2,self.trans1,self.trans2,self.trans3,self.trans4]:
            if isinstance(i,nn.Conv2d):
                nn.init.kaiming_normal_(
                    i.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(i.bias, 0)
            else:
                nn.init.normal_(i.weight, 0, 0.001)
                nn.init.constant_(i.bias, 0)

    @force_fp32(apply_to=('num_out', 'num_target',))
    def loss(self,num_out,num_target): 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        num_out=num_out.float().to(device)
        num_target=num_target.float().to(device)
        
        loss_as = self.count_loss(num_out,num_target)
        return loss_as

    def get_target(self,gt_bboxes):
        list = []
        for i in gt_bboxes:
            list.append(len(i))
        num_target = torch.tensor(list).view(len(gt_bboxes),-1)
        return num_target



if __name__=='__main__':
    x = (torch.rand(2,256,200,272),torch.rand(2,256,100,136),torch.rand(2,256,50,68),torch.rand(2,256,25,34),)
    model = asHead()
    print(model.count_loss)
    y = model(x)
    print(y)