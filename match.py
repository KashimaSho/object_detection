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
  A = box_a.size(0)
  B = box_b.size(0)