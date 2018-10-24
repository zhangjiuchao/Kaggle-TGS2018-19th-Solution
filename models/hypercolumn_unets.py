import os
import sys

from common import *
from .senet import *
import torchvision

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = SynchronizedBatchNorm2d(out_channels)

    
    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class GridAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, sub_sample_factor=(1,1)):
        super(GridAttentionBlock2D, self).__init__()

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        self.conv1 = ConvBn2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=sub_sample_factor, 
                                stride=sub_sample_factor, padding=0, bias=False)
        self.phi = nn.Conv2d(self.gating_channels, self.inter_channels, kernel_size=1, 
                                stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        # sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=True)
        y = sigm_psi_f.expand_as(x) * x

        return self.conv1(y)



class SpatialChannelSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SpatialChannelSE, self).__init__()

        self.fc1 = nn.Conv2d(channels, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channels, kernel_size=1, padding=0)
        self.fc3 = nn.Conv2d(channels, 1,         kernel_size=1, padding=0)

    def forward(self, x):
        z1 = F.adaptive_avg_pool2d(x, 1)
        z1 = self.fc1(z1)
        z1 = F.relu(z1, inplace=True)
        z1 = self.fc2(z1)
        g1 = F.sigmoid(z1)

        z2 = self.fc3(x)
        g2 = F.sigmoid(z2)

        x = g1*x + g2*x
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.scSE = SpatialChannelSE(out_channels)

    def forward(self, x, z=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        if z is not None:
            x = torch.cat([x, z], 1)
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)

        x = self.scSE(x)
        return x


class HyperAttResnet34(nn.Module):
    def __init__(self):
        super(HyperAttResnet34, self).__init__()
        self.encoder = torchvision.models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
        )
        self.encoder2 = self.encoder.layer1  # 64
        self.encoder3 = self.encoder.layer2  #128
        self.encoder4 = self.encoder.layer3  #256
        self.encoder5 = self.encoder.layer4  #512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # Attention Maps
        self.compatibility_score1 = GridAttentionBlock2D(64, 256)
        self.compatibility_score2 = GridAttentionBlock2D(128, 256)
        self.compatibility_score3 = GridAttentionBlock2D(256, 256)
        self.compatibility_score4 = GridAttentionBlock2D(512, 256)

        self.decoder4 = Decoder(256 + 128, 256, 64)
        self.decoder3 = Decoder(64 + 64, 128, 64)
        self.decoder2 = Decoder(64 + 32, 64, 64)
        self.decoder1 = Decoder(64, 64, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)

        x = self.conv1(x)

        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)

        f5 = self.compatibility_score4(e5, f)
        f4 = self.compatibility_score3(e4, f)
        f3 = self.compatibility_score2(e3, f)
        f2 = self.compatibility_score1(e2, f)

        d4 = self.decoder4(f5, f4)  #;print("d4 size: ", d4.size())
        d3 = self.decoder3(d4, f3)  #;print("d3 size: ", d3.size())
        d2 = self.decoder2(d3, f2)  #;print("d2 size: ", d2.size())
        d1 = self.decoder1(d2)      #;print("d1 size: ", d1.size())

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
        ], 1)

        f = F.dropout2d(f, p=0.5)

        logit = self.logit(f)   #;print("logit size: ", logit.size())
        return logit

    def criterion(self, logit, truth ):
    
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)

        loss = lovasz_hinge(logit, truth)
        return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

    def load_pretrain(self, pretrain_file):
        
        self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))



class HEDResnet34(nn.Module):

    def __init__(self):
        super(HEDResnet34, self).__init__()

        self.encoder = torchvision.models.resnet34()

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
        )
        self.encoder2 = self.encoder.layer1  # 64
        self.encoder3 = self.encoder.layer2  #128
        self.encoder4 = self.encoder.layer3  #256
        self.encoder5 = self.encoder.layer4  #512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.decoder5 = Decoder(256+512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 64, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)

        x = self.conv1(x)

        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)

        d5 = self.decoder5(f,  e5)  #;print("d5 size: ", d5.size())
        d4 = self.decoder4(d5, e4)  #;print("d4 size: ", d4.size())
        d3 = self.decoder3(d4, e3)  #;print("d3 size: ", d3.size())
        d2 = self.decoder2(d3, e2)  #;print("d2 size: ", d2.size())
        d1 = self.decoder1(d2)      #;print("d1 size: ", d1.size())
        
        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False)
        d4 = F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False)
        d5 = F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)

        f = torch.cat([d1, d2, d3, d4, d5,
        ], 1)

        f = F.dropout2d(f, p=0.5)

        logit = self.logit(f)   #;print("logit size: ", logit.size())
        return logit


    def criterion(self, logit, truth ):

        # #loss = PseudoBCELoss2d()(logit, truth)
        # #loss = FocalLoss2d()(logit, truth, type='sigmoid')
        # loss = RobustFocalLoss2d(size_average=False)(logit, truth, type='sigmoid')
    
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)

        loss = lovasz_hinge(logit, truth)
        return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

    def load_pretrain(self, pretrain_file):
        
        self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))


## SE-Resnet #####################
class DeepHedSENeXt50(nn.Module):
    def __init__(self):
        super(DeepHedSENeXt50, self).__init__()

        self.encoder = se_resnext50_32x4d()

        self.layer0 = nn.Sequential(
            self.encoder.layer0[0],
            self.encoder.layer0[1],
            self.encoder.layer0[2]
        )
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.decoder5 = Decoder(512+2048, 1024, 128)
        self.decoder4 = Decoder(128 + 1024, 512, 128)
        self.decoder3 = Decoder(128 + 512, 256, 128)
        self.decoder2 = Decoder(128 + 256, 128, 128)
        self.decoder1 = Decoder(128, 128, 128)

        self.cls1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.cls2 = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.logit1 = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.logit2 = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.logit3 = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.logit4 = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.logit5 = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.logit = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)

        x = self.layer0(x)

        e2 = self.layer1(x)
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4)

        f = self.center(e5)

        d5 = self.decoder5(f,  e5)  #;print("d5 size: ", d5.size())
        d4 = self.decoder4(d5, e4)  #;print("d4 size: ", d4.size())
        d3 = self.decoder3(d4, e3)  #;print("d3 size: ", d3.size())
        d2 = self.decoder2(d3, e2)  #;print("d2 size: ", d2.size())
        d1 = self.decoder1(d2)      #;print("d1 size: ", d1.size())
        
        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False)
        d4 = F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False)
        d5 = F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)

        f_cls = F.adaptive_avg_pool2d(f, 1)
        f_cls = F.relu(self.cls1(f_cls), inplace=True)
        cls_logit = self.cls2(f_cls)

        logit1 = self.logit1(d1)
        logit2 = self.logit2(d2)
        logit3 = self.logit3(d3)
        logit4 = self.logit4(d4)
        logit5 = self.logit5(d5)

        f = torch.cat([d1, d2, d3, d4, d5,
            F.upsample(f_cls, scale_factor=128, mode='bilinear', align_corners=False)
        ], 1)

        f = F.dropout2d(f, p=0.5)

        logit = self.logit(f)   #;print("logit size: ", logit.size())
        return logit, cls_logit, logit1, logit2, logit3, logit4, logit5

    def criterion(self, logit, truth ):
        batch_size = truth.size(0)
        mask_sum = torch.sum(truth.view(batch_size, -1), 1).view(-1)
        nonempty_index = (mask_sum > 0).nonzero().view(-1)
    
        loss = weighted_binary_cross_entropy_with_logits(logit[1], (mask_sum>0).float()) * 0.05
        truth = truth.squeeze(1)
        lo = logit[0].squeeze(1)
        loss += lovasz_hinge(lo, truth)

        if nonempty_index.size(0) >0:
            nonempty_truth = torch.index_select(truth, 0, nonempty_index)
            for i in range(2, 7):
                lo = torch.index_select(logit[i], 0, nonempty_index)
                loss += RobustFocalLoss2d(size_average=True)(lo, nonempty_truth, type='sigmoid') * 0.1

        return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

class HedSENet50(nn.Module):
    def __init__(self):
        super(HedSENet50, self).__init__()

        self.encoder = se_resnext50_32x4d()

        self.layer0 = nn.Sequential(
            self.encoder.layer0[0],
            self.encoder.layer0[1],
            self.encoder.layer0[2]
        )
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.decoder5 = Decoder(512+2048, 1024, 128)
        self.decoder4 = Decoder(128 + 1024, 512, 128)
        self.decoder3 = Decoder(128 + 512, 256, 128)
        self.decoder2 = Decoder(128 + 256, 128, 128)
        self.decoder1 = Decoder(128, 128, 128)

        self.logit = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    
    def forward(self, x):

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)

        x = self.layer0(x)      #;print("x size: ", x.size())

        e1 = self.layer1(x)     #;print("e1 size: ", e1.size())
        e2 = self.layer2(e1)    #;print("e2 size: ", e2.size())
        e3 = self.layer3(e2)    #;print("e3 size: ", e3.size())
        e4 = self.layer4(e3)    #;print("e4 size: ", e4.size())

        f = self.center(e4)

        d5 = self.decoder5(f, e4)
        d4 = self.decoder4(d5, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False)
        d4 = F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False)
        d5 = F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)

        f = torch.cat([d1, d2, d3, d4, d5], 1)

        f = F.dropout2d(f, p=0.5)

        logit = self.logit(f)   #;print(logit.size())

        return logit

    def criterion(self, logit, truth ):

        #loss = PseudoBCELoss2d()(logit, truth)
        #loss = FocalLoss2d()(logit, truth, type='sigmoid')
        # loss = RobustFocalLoss2d()(logit, truth, type='sigmoid')
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)

        loss = lovasz_hinge(logit, truth)
        
        return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


class HedSEResXt101(nn.Module):
    def __init__(self):
        super(HedSEResXt101, self).__init__()
        self.encoder = se_resnext101_32x4d()

        self.layer0 = nn.Sequential(
            self.encoder.layer0[0],
            self.encoder.layer0[1],
            self.encoder.layer0[2]
        )
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.decoder5 = Decoder(512+2048, 1024, 128)
        self.decoder4 = Decoder(128 + 1024, 512, 128)
        self.decoder3 = Decoder(128 + 512, 256, 128)
        self.decoder2 = Decoder(128 + 256, 128, 128)
        self.decoder1 = Decoder(128, 128, 128)

        self.logit = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )


    def forward(self, x):

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)

        x = self.layer0(x)      #;print("x size: ", x.size())

        e1 = self.layer1(x)     #;print("e1 size: ", e1.size())
        e2 = self.layer2(e1)    #;print("e2 size: ", e2.size())
        e3 = self.layer3(e2)    #;print("e3 size: ", e3.size())
        e4 = self.layer4(e3)    #;print("e4 size: ", e4.size())

        f = self.center(e4)

        d5 = self.decoder5(f, e4)
        d4 = self.decoder4(d5, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False)
        d4 = F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False)
        d5 = F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)


        f = torch.cat([d1, d2, d3, d4, d5], 1)

        f = F.dropout2d(f, p=0.5)

        logit = self.logit(f)   #;print(logit.size())

        return logit

    def criterion(self, logit, truth ):

        #loss = PseudoBCELoss2d()(logit, truth)
        #loss = FocalLoss2d()(logit, truth, type='sigmoid')
        # loss = RobustFocalLoss2d()(logit, truth, type='sigmoid')
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)

        loss = lovasz_hinge(logit, truth)
        
        return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

class HedSEResXt154(nn.Module):
    def __init__(self):
        super(HedSEResXt154, self).__init__()
        self.encoder = senet154()

        self.layer0 = nn.Sequential(
            *self.encoder.layer0[:9]            
        )
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.decoder5 = Decoder(512+2048, 1024, 128)
        self.decoder4 = Decoder(128 + 1024, 512, 128)
        self.decoder3 = Decoder(128 + 512, 256, 128)
        self.decoder2 = Decoder(128 + 256, 128, 128)
        self.decoder1 = Decoder(128, 128, 128)

        self.logit = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0)
        )


    def forward(self, x):

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)

        x = self.layer0(x)      #;print("x size: ", x.size())

        e1 = self.layer1(x)     #;print("e1 size: ", e1.size())
        e2 = self.layer2(e1)    #;print("e2 size: ", e2.size())
        e3 = self.layer3(e2)    #;print("e3 size: ", e3.size())
        e4 = self.layer4(e3)    #;print("e4 size: ", e4.size())

        f = self.center(e4)

        d5 = self.decoder5(f, e4)
        d4 = self.decoder4(d5, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False)
        d4 = F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False)
        d5 = F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)

        f = torch.cat([d1, d2, d3, d4, d5], 1)

        f = F.dropout2d(f, p=0.5)

        logit = self.logit(f)   #;print(logit.size())

        return logit

    def criterion(self, logit, truth ):

        #loss = PseudoBCELoss2d()(logit, truth)
        #loss = FocalLoss2d()(logit, truth, type='sigmoid')
        # loss = RobustFocalLoss2d()(logit, truth, type='sigmoid')
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)

        loss = lovasz_hinge(logit, truth)
        
        return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


