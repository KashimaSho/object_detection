import torch #とりあえず
def point_form(boxes):
  '''
  Parameter:
    boxes(Tensor): information of DBox(box_num, 4)
  Return:
    boxes(Tensor): information of BBox(box_num, 4)
  '''
  
  return torch.cat(
    (boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2),
    1
  )
  
def intersect(box_a, box_b):
  '''
  Parameter:
    box_a(Tensor): Coordinates of BBox(box_num, 4)
    box_b(Tensor): Coordinates of BBox(box_num, 4)
  Return:
    intersection of box_a and box_b
  '''
  # get box num
  A = box_a.size(0)
  B = box_b.size(0)
  
  # caluculate intersection of box_a and box_b
  max_xy = torch.min(
    # tensor.unsqueeze(dim)でdim次元にサイズ1の次元を挿入
    # A(0)vsB(0) ~ A(n)vsB(n)まですべての組み合わせでmin(xmax, ymax)を求める
    box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
    box_b[:, 2:].unsqueeze(0).expand(A, B, 2)
  )
  min_xy = torch.max(
    # A(0)vsB(0) ~ A(n)vsB(n)まですべての組み合わせでmax(xmin, ymin)を求める
    box_a[:, :2].unsqueeze(1).expand(A, B, 2),
    box_b[:, :2].unsqueeze(0).expand(A, B, 2)
  )
  
  # torch.clamp(input, min, max)でmin<=input<=maxになるように調整
  inter = torch.clamp((max_xy - min_xy), min=0)
  return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
  '''
  Parameter:
    box_a(Tensor): Coordinates of BBox(box_num, 4)
    box_b(Tensor): Coordinates of BBox(box_num, 4)
  Return:
    IoU of box_a and box_b
  '''
  
  inter = intersect(box_a, box_b)
  area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
  area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
  
  union = area_a + area_b - inter
  return inter / union

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
  '''
  教師データloc, confを作成
  Parameter:
    threshold(float): threshold of IoU
    truths(Tensor): coordinates of BBox in minibatch (BBox_num, 4)
    priors(Tensor): information of DBox (8732, 4)
    variances(list): [0.1, 0.2] to caluculate offset value of DBox
    labels(list[int]): label list (BBox1, BBox2, ...)
    loc_t(Tensor): Tensor of BBox label which is the closest to each DBox (batch_size, 8732, 4)
    conf_t(Tensor): Tensor of BBox label which is the closest to each DBox (batch_size, 8732)
    idx(int): minibatch index
  Return: None
  '''
  
  overlaps = jaccard(truths, point_form(priors))
  best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
  best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
  best_truth_idx.squeeze_(0)
  best_truth_overlap.squeeze_(0)
  
  best_prior_idx.squeeze_(1)
  best_prior_overlap.squeeze_(1)
  
  best_truth_overlap.index_fill_(
    0,
    best_prior_idx,
    2
  )
  
  for j in range(best_prior_idx.size(0)):
    best_truth_idx[best_prior_idx[j]] = j
  
  matches = truths[best_truth_idx]
  conf = labels[best_truth_idx] + 1
  conf[best_truth_overlap < threshold] = 0
  
  # defined below
  loc = encode(
    matches,
    priors,
    variances
  )
  
  loc_t[idx] = loc
  conf_t[idx] = conf

def encode(matched, priors, variances):
  '''
  DBoxの情報[cx, cy, w, h]をDBoxのオフセット情報[Δx, Δy, Δw, Δh]に変換する
  Parameter:
    matched(Tensor): BBox annotation data which is matched to DBox
    priors(Tensor): information of DBox (8732, 4)
    variances(list[float]): coefficient used to caluculate offset value of DBox
  Return:
    offset information of DBox (DBox_num, 4)
  '''
  
  g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
  g_cxcy /= (variances[0] * priors[:, 2:])
  
  g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
  g_wh = torch.log(g_wh) / variances[1]
  
  return torch.cat([g_cxcy, g_wh], 1)
  