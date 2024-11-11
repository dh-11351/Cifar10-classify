import os
# from scipy.misc import imsave
from imageio import imsave


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


filename = 'D:/shenduxuexi/keshe/cifar-10-python/cifar-10-batches-py'   # 图片的路径
meta = unpickle(filename + '/batches.meta')
label_name = meta[b'label_names']

for i in range(len(label_name)):
    file = label_name[i].decode()
    path = 'D:/shenduxuexi/keshe/cifar-10-python/cifar-10-batches-py/train/' + file
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

for i in range(1, 6):
    content = unpickle(filename + '/data_batch_' + str(i))  # 解压后的每个data_batch_
    print('load data...')
    print(content.keys())
    print('tranfering data_batch' + str(i))

    for j in range(10000):
        img = content[b'data'][j]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img_name = 'D:/shenduxuexi/keshe/cifar-10-python/cifar-10-batches-py/train/' + label_name[
            content[b'labels'][j]].decode() + '/batch_' + str(i) + '_num_' + str(
            j) + '.jpg'
        imsave(img_name, img)


'''
filename = 'D:/shenduxuexi/keshe/cifar-10-python/cifar-10-batches-py'  # 图片的路径
meta = unpickle(filename + '/batches.meta')
label_name = meta[b'label_names']

for i in range(len(label_name)):
    file = label_name[i].decode()
    path = 'D:/shenduxuexi/keshe/cifar-10-python/cifar-10-batches-py/test/' + file
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

content = unpickle(filename + '/test_batch' )  # 解压后的每个data_batch_
print('load data...')
print(content.keys())
print('tranfering test_batch')
for j in range(10000):
        img = content[b'data'][j]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img_name = 'D:/shenduxuexi/keshe/cifar-10-python/cifar-10-batches-py/test/' + label_name[
            content[b'labels'][j]].decode() + '/batch_'  + '_num_' + str(
            j) + '.jpg'
        imsave(img_name, img)
'''