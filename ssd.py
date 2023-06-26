import torch
import torch.nn.init as init
import torch.nn as nn
from itertools import product as product
from math import sqrt as sqrt
from torch.autograd import Function
import torch.nn.functional as F

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
    dbox_num(int list): out1~out6それぞれに用意されるデフォルトボックスの数
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
    dbox_num(int list): out1~out6それぞれに用意されるデフォルトボックスの数
    class_num(int): クラス数
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
      input_channels(int): num of input channels = num of output channels at vgg4
      scale(int): initial value of weight
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
  
  
class DBox(object):
  '''
  8732個のDBoxの(x, y, width, height)を生成するクラス
  Attribute:
    image_size(int): イメージサイズ
    feature_maps(list): out1~out6の特徴量マップリスト [38, 19, 10, 5, 3, 1]
    num_priors(int): feature_mapsの要素数 6
    steps(list): DBoxのサイズのリスト 
    min_sizes(list): 小さい正方形のDBoxのサイズ
    max_sizes(list): 大きい正方形のDBoxのサイズ
    aspect_ratios(list): 長方形のDBoxのアスペクト比
  '''
  
  def __init__(self, cfg):
    super(DBox, self).__init__()
    
    self.image_size = cfg['input_size']
    self.feature_maps = cfg['feature_maps']
    self.num_priors = len(cfg['feature_maps'])
    self.steps = cfg['steps']
    self.min_sizes = cfg['min_sizes']
    self.max_sizes = cfg['max_sizes']
    self.aspect_ratios = cfg['aspect_ratios']
  
  def make_dbox_list(self):
    '''
    Return:
      (Tensor)DBoxの[cx, cy, width, height]を格納した(8732, 4)のテンソル
    '''
    mean = []
    #feature_maps = [38, 19, 10, 5, 3, 1]
    for k, f in enumerate(self.feature_maps):
      #(i,j)はf=38の場合, (0,0), (0,1), ..., (0,37)~(37,0), (37,1), ..., (37,37)
      for i, j in product(range(f), repeat=2):
        f_k = self.image_size / self.steps[k]
        cx = (j + 0.5) / f_k
        cy = (i + 0.5) / f_k
        
        #min_sizes = [30, 60, 111, 162, 213, 264] / 300
        s_k = self.min_sizes[k] / self.image_size
        mean += [cx, cy, s_k, s_k]
        
        #max_sizes = [45, 99, 153, 207, 261, 315] / 300
        s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
        mean += [cx, cy, s_k_prime, s_k_prime]
        
        for ar in self.aspect_ratios[k]:
          mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
          mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        
    output = torch.Tensor(mean).view(-1, 4)
    output.clamp_(max=1, min=0)
    
    return output
  
  
def decode(loc, dbox_list):
  '''
  Decode default box to bounding box
  
  Parameter:
    loc(Tensor): (8732, 4) output of loc which means offset of DBox (Δcx, Δcy, Δwidth, Δheight)
    dbox_list(Tensor): (8732, 4) tensor of DBox (cx_d, cy_d, width_d, height_d)
  Return:
    boxes(Tensor): (8732, 4) tensor of BBox(xmin, ymin, xmax, ymax)
  '''
  
  boxes = torch.cat((
    # cx = cx_d + 0.1 * Δcx * w_d
    # cy = cy_d + 0.1 * Δcy * h_d
    dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
    # w = width_d * exp(0.2 * Δwidth)
    # h = height_d * exp(0.2 * Δheight)
    dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)
  ), dim=1)
  
  boxes[:, :2] -= boxes[:, 2:] / 2 # Add width and height to center coordinates in order to convert xmin and ymin
  boxes[:, 2:] += boxes[:, :2] # Add xmin and ymin to width and height in order to convert xmax and ymax
  
  return boxes
  

def nonmaximum_suppress(boxes, scores, overlap=0.5, top_k=200):
  '''
  1つの物体に対して1つのBBoxだけ残す
  21クラスすべてにこの処理を施す
  確信度scoresとIoUを利用してそれぞれの物体のBBoxを1つだけ決める
  
  Parameter:
    boxes(Tensor): tensor of BBox (confident score > 0.01)
    scores(Tensor): confident score ( > 0.01) from "conf"
    overlap(float): the threshold of IoU to determine which are same objects
    top_k(int): num of extracting sample (confident score > 0.01)
  Return:
    keep(Tensor): index of BBox
    count(int): num of BBox
  '''
  
  # initialization
  count = 0
  keep = scores.new(scores.size(0)).zero_().long()
  
  # caluculate area of BBox
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]
  area = torch.mul(x2-x1, y2-y1)
  
  #make empty tensors
  tmp_x1 = boxes.new()
  tmp_y1 = boxes.new()
  tmp_x2 = boxes.new()
  tmp_y2 = boxes.new()
  tmp_w = boxes.new()
  tmp_h = boxes.new()
  
  # extract top_k num boxes (confident score > 0.01)
  v, idx = scores.sort(0)
  idx = idx[-top_k:]
  
  while idx.numel() > 0: # numel()は要素数, dim()は次元数, size()は形状
    i = idx[-1]
    keep[count] = i
    count += 1
    
    # break if it is last BBox
    if idx.size(0) == 1:
      break
    
    # index_select(input_tensor, dim, index_element, output_tensor)
    idx = idx[:-1]
    torch.index_select(x1, 0, idx, out=tmp_x1)
    torch.index_select(y1, 0, idx, out=tmp_y1)
    torch.index_select(x2, 0, idx, out=tmp_x2)
    torch.index_select(y2, 0, idx, out=tmp_y2)
    
    tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
    tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
    tmp_x2 = torch.clamp(tmp_x2, min=x2[i])
    tmp_y2 = torch.clamp(tmp_y2, min=y2[i])
    
    tmp_w.resize_as_(tmp_x2)
    tmp_h.resize_as_(tmp_y2)
    tmp_w = tmp_x2 - tmp_x1
    tmp_h = tmp_y2 - tmp_y1
    
    # calculate IoU
    inter = tmp_w * tmp_h
    rem_areas = torch.index_select(area, 0, idx)
    union = (rem_areas - inter) + area[i]
    IoU = inter / union
    
    # remove BBox (IoU < overlap)
    idx = idx[IoU.le(overlap)]
    
  return keep, count
    

class Detect(Function):
  '''
  推論時のforward処理を実装
  Attribute:
    softmax: torch.nn.Softmax
    conf_thresh: threshold to extract BBox
    top_k: num of BBox by NMS
    nms_thresh: threshold of IoU
  '''
  
  @staticmethod
  def forward(ctx, loc_data, conf_data, dbox_list):
    '''
    Parameter:
      loc_data(Tensor): output from loc network
      conf_data(Tensor): output from conf network
      dbox_list(Tensor): information of dboxes
    Return:
      output(Tensor): (batch_num, 21, 200, 5) which means (batch_data_idx, class_idx, BBox_idx, [conf, xmin, ymin, xmax, ymax])
    '''
    ctx.softmax = nn.Softmax(dim=-1)
    ctx.conf_thresh = 0.01
    ctx.top_k = 200
    ctx.nms_thresh = 0.45
    
    batch_num = loc_data.size(0)
    classes_num = conf_data.size(2)
    conf_data = ctx.softmax(conf_data)
    conf_preds = conf_data.transpose(2, 1)
    output = torch.zeros(batch_num, classes_num, ctx.top_k, 5)
    
    for i in range(batch_num):
      decoded_boxes = decode(loc_data[i], dbox_list)
      conf_scores = conf_preds[i].clone()
      
      for cl in range(1, classes_num):
        #conf_thresholdを超えていればTrue, そうでなければFalse
        c_mask = conf_scores[cl].gt(ctx.conf_thresh)
        scores = conf_scores[cl][c_mask]
        if scores.nelement() == 0:
          continue
        
        # c_maskの形状を(8732,)から(8732, 4)にする
        l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
        boxes = decoded_boxes[l_mask].view(-1, 4)
        
        ids, count = nonmaximum_suppress(boxes, scores, ctx.nms_thresh, ctx.top_k)
        output[i, cl, :count] = torch.cat(
          (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]),
          dim=1
        )
    return output
    
    
class SSD(nn.Module):
    '''
    SSDモデルを生成
    Attribute:
      phase(str): train or test
      classes_num(int): class num
      vgg(object): vgg network
      extras(object): extras network
      L2Norm(object): L2norm layer
      loc(object): loc network
      conf(object): conf network
      dbox_list(Tensor): DBox[cx, cy, width, height]
      detect(object): objects to execute forward() in Detect class
    '''
    
    def __init__(self, phase, cfg):
      super(SSD, self).__init__()
      
      self.phase = phase
      self.classes_num = cfg['classes_num']
      self.vgg = make_vgg()
      self.extras = make_extras()
      self.L2Norm = L2Norm()
      self.loc = make_loc(cfg['dbox_num'])
      self.conf = make_conf(cfg['classes_num'], cfg['dbox_num'])
      
      dbox = DBox(cfg=cfg)
      self.dbox_list = dbox.make_dbox_list()
      
      if phase == 'test':
        self.detect = Detect.apply
    
    def forward(self, x):
      '''
      Parameter:
        x(Tensor): (batch_size, 3, 300, 300)
      Return:
        if phase == 'test':
          1枚の画像に対するBBoxの情報を格納(batch_size, 21, 200, 5) 
        elif phase == 'train':
          (loc, conf, dbox_list)を格納したタプル
          loc(batch_size, 8732, 4)
          conf(batch_size, 8732, 21)
          dbox_list(8732, 4)
      '''
      out_list = list()
      loc = list()
      conf = list()
      
      # get out1
      # from vgg1 to vgg4
      for k in range(23):
        x = self.vgg[k](x)
      out1 = self.L2Norm(x)
      out_list.append(out1)
      
      # get out2
      # from vgg4 to vgg6
      for k in range(23, len(self.vgg)):
        x = self.vgg[k](x)
      out_list.append(x)
      
      # get out3 ~ out6
      #from extras1 to extras4
      for k,v in enumerate(self.extras):
        x = F.relu(v(x), inplace=True)
        # each extras has 2 layers
        # add to out_list after every odd layer
        if k%2 == 1:
          out_list.append(x)
      
      for (x, l, c) in zip(out_list, self.loc, self.conf):
        #loc networkにout1 ~ out6を入力
        #形状を(batch_size, offset*DBox_num, height, width)から(batch_size, height, width, offset*DBox_num)に変換
        #torch.contiguous()でメモリ上に要素を連続的に配置し直してview()関数を適用できるようにする
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        
        #conf networkにout1 ~ out6を入力
        #形状を(batch_size, classes_num*DBox_num, height, width)から(batch_size, height, width, classes_num*DBox_num)に変換
        #torch.contiguous()でメモリ上に要素を連続的に配置し直してview()関数を適用できるようにする
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
      loc = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1)
      conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)
      
      loc = loc.view(loc.size(0), -1, 4)
      conf = conf.view(conf.size(0), -1, self.classes_num)
      output = (loc, conf, self.dbox_list)
      
      if self.phase == 'test':
        return self.detect(output[0], output[1], output[2])
      else:
        return output
        
        
        
        
        
        
        
      
      
          
    