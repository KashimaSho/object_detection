import random
def __call__(self, image, boxes=None, labels=None):
    if random.randint(2):
        delta = random.uniform(-self.delta, self.delta)
        image += delta
    return image, boxes, labels