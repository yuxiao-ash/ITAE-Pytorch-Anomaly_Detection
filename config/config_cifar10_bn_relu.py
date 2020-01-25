dataset = 'cifar10'
dataroot = ''
isize = 32
nc = 3
bilinear = True
transformations = 4

manualseed = -1
workers = 4
epoch_size = 500 // transformations
milestone = [int(0.1 * i * epoch_size) for i in range(1, 10)]
batchsize = 32
best_auroc = 0.1
show_frequent = 50
lr = 0.1

work_root = './work_dir'
work_dir = work_root + '/cifar10_bn_relu_car_seed-1'
resume_from = None

normal_class = 'car'
"""
classes of cifar 10:
    'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
"""
exp = 'ITAE experiment on cifar10 dataset \n' \
      'set ' + normal_class + ' to be normal class and other classes to be abnormal class\n' \
      'manualseed in data.py is set to -1, not 10 any more\n'