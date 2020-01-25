dataset = 'mnist'
dataroot = ''
isize = 28
nc = 1
bilinear = True
transformations = 4

manualseed = -1
workers = 4
epoch_size = 500 // 1
milestone = [int(0.1 * i * epoch_size) for i in range(1, 10)]
batchsize = 32
best_auroc = 0.1
show_frequent = 50
lr = 0.1

work_root = './work_dir'
work_dir = work_root + '/mnist_normal_3'
resume_from = None

normal_class = '3'
"""
classes of mnist: 0 ~ 9
"""
exp = 'ITAE experiment on mnist dataset\n' \
      'set ' + normal_class + ' to be normal class and other classes to be abnormal class\n'