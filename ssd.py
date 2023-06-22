import torch
import torch.nn.init as init
import torch.nn as nn

def make_vgg():
  '''
  Return:
    (nn.ModuleList): vgg module list
  '''
  
  vgg_layers = []
  in_channels = 3
  
  #M, MCはプーリング層
  cfg = [64, 64, 'M', #vgg1
         128, 128, 'M', #vgg2
         256, 256, 256, 'MC', #vgg3
         512, 512, 512, 'M', #vgg4
         512, 512, 512 #vgg5
         ]
  
  for v in cfg:
    if v == 'M':
      vgg_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    elif v == 'MC':
      #ceil_mode=Trueで出力される特徴量マップのサイズを切り上げる(37.5->38)
      vgg_layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
    else:
      conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
      #inplace=TrueにすることでReLUへの入力値を保持せずにメモリを節約する
      vgg_layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  
  pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
  #vgg6 dilationにより大域的な情報を取り込む
  conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
  conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
  
  vgg_layers += [pool5, 
             conv6, nn.ReLU(inplace=True),
             conv7, nn.ReLU(inplace=True)]
  
  return nn.ModuleList(vgg_layers)


def make_extras():
  '''
  Return:
    (nn.ModuleList): extras module list
  '''
  
  extras_layers = []
  in_channels = 1024
  
  cfg = [256, 512, #extras1
         128, 256, #extras2
         128, 256, #extras3
         128, 256, #extras4
         ]
  
  #extras1
  extras_layers += [nn.Conv2d(in_channels=in_channels, out_channels=cfg[0], kernel_size=1)]
  extras_layers += [nn.Conv2d(in_channels=cfg[0], out_channels=cfg[1], kernel_size=3, stride=2, padding=1)]
  
  #extras2
  extras_layers += [nn.Conv2d(in_channels=cfg[1], out_channels=cfg[2], kernel_size=1)]
  extras_layers += [nn.Conv2d(in_channels=cfg[2], out_channels=cfg[3], kernel_size=3, stride=2, padding=1)]
  
  #extras3
  extras_layers += [nn.Conv2d(in_channels=cfg[3], out_channels=cfg[4], kernel_size=1)]
  extras_layers += [nn.Conv2d(in_channels=cfg[4], out_channels=cfg[5], kernel_size=3)]
  
  #extras4
  extras_layers += [nn.Conv2d(in_channels=cfg[5], out_channels=cfg[6], kernel_size=1)]
  extras_layers += [nn.Conv2d(in_channels=cfg[6], out_channels=cfg[7], kernel_size=3)]
  
  return nn.ModuleList(extras_layers)


def make_loc(dbox_num=[4, 6, 6, 6, 4, 4]):
  '''
  Prameter:
    dbox_num(Int list): out1~out6それぞれに用意されるデフォルトボックスの数
  Return:
    (nn.ModuleList): loc module list
  '''
  
  loc_layers = []
  #from out1(vgg4)
  loc_layers += [nn.Conv2d(in_channels=512, out_channels=dbox_num[0]*4, kernel_size=3, padding=1)]
  #from out2(vgg6)
  loc_layers += [nn.Conv2d(in_channels=1024, out_channels=dbox_num[1]*4, kernel_size=3, padding=1)]
  #from out3(extras1)
  loc_layers += [nn.Conv2d(in_channels=512, out_channels=dbox_num[2]*4, kernel_size=3, padding=1)]
  #from out4(extras2)
  loc_layers += [nn.Conv2d(in_channels=256, out_channels=dbox_num[3]*4, kernel_size=3, padding=1)]
  #from out5(extras3)
  loc_layers += [nn.Conv2d(in_channels=256, out_channels=dbox_num[4]*4, kernel_size=3, padding=1)]
  #from out6(extras4)
  loc_layers += [nn.Conv2d(in_channels=256, out_channels=dbox_num[5]*4, kernel_size=3, padding=1)]
  
  return nn.ModuleList(loc_layers)


def make_conf(class_num=21, dbox_num=[4, 6, 6, 6, 4, 4]):
  '''
  Prameter:
    dbox_num(Int list): out1~out6それぞれに用意されるデフォルトボックスの数
    class_num(Int): クラス数
  Return:
    (nn.ModuleList): conf module list
  '''
  
  conf_layers = []
  #from out1(vgg4)
  conf_layers += [nn.Conv2d(in_channels=512, out_channels=dbox_num[0]*class_num, kernel_size=3, padding=1)]
  #from out2(vgg6)
  conf_layers += [nn.Conv2d(in_channels=1024, out_channels=dbox_num[1]*class_num, kernel_size=3, padding=1)]
  #from out3(extras1)
  conf_layers += [nn.Conv2d(in_channels=512, out_channels=dbox_num[2]*class_num, kernel_size=3, padding=1)]
  #from out4(extras2)
  conf_layers += [nn.Conv2d(in_channels=256, out_channels=dbox_num[3]*class_num, kernel_size=3, padding=1)]
  #from out5(extras3)
  conf_layers += [nn.Conv2d(in_channels=256, out_channels=dbox_num[4]*class_num, kernel_size=3, padding=1)]
  #from out6(extras4)
  conf_layers += [nn.Conv2d(in_channels=256, out_channels=dbox_num[5]*class_num, kernel_size=3, padding=1)]
  
  return nn.ModuleList(conf_layers)


class L2Norm(nn.Module):
  '''
  vgg4からの出力out1をL2Normで正規化する層
  Attribute:
    weight: parameters of L2Norm layer
    scale: initial value of weight
    eps: L2Normに加算する極小値
  '''
  
  def __init__(self, input_channels=512, scale=20):
    '''
    Parameter:
      input_channels(Int): num of input channels = num of output channels at vgg4
      scale(Int): initial value of weight
    '''
    
    super(L2Norm, self).__init__() #親のnn.Moduleのコンストラクタ__init__()を実行
    self.weight = nn.Parameter(torch.Tensor(input_channels))
    self.scale = scale
    self.reset_parameters() #defined below
    self.eps = 1e-10
    
  def reset_parameters(self):
    '''
    initialize all weight by value of scale
    '''
    init.constant_(self.weight, self.scale) #nn.initの関数 initilize Param1(Tensor) by Param2(val)
  
  def forward(self, x):
    '''
    Parameter:
      x(Tensor): output of vgg4 (batch_size, 5112, 38, 38)
    Return:
      4階のテンソル(batch_size, 512, 38, 38)
    '''
    
    #2乗して 和を取って 平方根取って 極小値を足す
    norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
    #同じセルのnormで割って正規化
    x = torch.div(x, norm)
    #4階のテンソルに変形してxと同じ形にする
    weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
    out = weights * x
    
    return out