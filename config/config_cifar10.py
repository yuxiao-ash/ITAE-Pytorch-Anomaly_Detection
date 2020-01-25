dataset = 'cifar10'
dataroot = ''
isize = 32
nc = 3
bilinear = True
transformations = 4

manualseed = 10
workers = 4
epoch_size = 500 // transformations
milestone = [int(0.1 * i * epoch_size) for i in range(1, 10)]
batchsize = 32
best_auroc = 0.1
show_frequent = 50
lr = 0.1

work_root = './work_dir'
work_dir = work_root + '/cifar10'
resume_from = None

exp = 'ITAE experiment on cifar10 dataset, ' \
      'set deer to be normal class and other classes to be abnormal class'
normal_class = 'deer'
"""
classes of cifar 10:
    'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
"""