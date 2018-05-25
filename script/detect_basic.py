# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/03/10
# description: start script
from __future__ import print_function
import sys
import argparse
import os
import platform
import collections
import random
import numpy
import multiprocessing as mp
from multiprocessing.sharedctypes import Array
from ctypes import c_double, cast, POINTER
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

if 'Windows' in platform.platform():
    store_dir = 'E:\Github\\table-detection\\'
elif 'Linux' in platform.platform():
    data_dir = '/home/ZhangJiexin/yolo-tensorflow/'


def main(method='train', gpus=''):
    from src.data.data_basic import Processor
    from src.network.network_basic import Network
    from src.model.model_basic import Model
   
    option = collections.OrderedDict()
    option['batch_size'] = 32
    option['image_x_size'] = 448
    option['image_y_size'] = 448
    option['n_channel'] = 3
    option['n_classes'] = 20
    option['cell_x_size'] = 7
    option['cell_y_size'] = 7
    option['pool_mode'] = 'max'
    option['n_boxes'] = 5
    option['n_processes'] = 1
    option['max_objects'] = 30
    option['n_iter'] = 600000000
    option['buffer_size'] = 5
    option['gpus'] = gpus
    option['n_gpus'] = len(gpus.split(',')) if len(gpus.split(',')) != 0 else 1
    option['is_multigpu'] = True if option['n_gpus'] > 1 else False
    option['is_valid'] = False
    option['noobject_scale'] = 0.5
    option['object_scale'] = 1
    option['coord_scale'] = 5
    option['class_scale'] = 1
    option['is_weight_decay'] = False
    option['weight_decay'] = 1e-3
    option['learning_rate'] = 1e-4
    option['is_lr_decay'] = False
    option['train_data'] = 'voc'
    option['test_data'] = 'voc'
    option['seq'] = 'voc-v1'
    option['model'] = 'model_best.ckpt'
    option['update_function'] = 'momentum'
    option['is_observe'] = True
    
    # 打印option
    print()
    for key in option:
        print('%-20s' % (key), '= {}'.format(option[key]))
    print()
    
    processor = Processor(
        image_x_size = option['image_x_size'], 
        image_y_size = option['image_y_size'], 
        max_objects = option['max_objects'], 
        n_classes = option['n_classes'],
        cell_x_size = option['cell_x_size'], 
        cell_y_size = option['cell_y_size'],
        n_boxes = option['n_boxes'],
        batch_size = option['batch_size'], 
        n_channel = option['n_channel'],
        n_processes = option['n_processes'], 
        n_iters = option['n_iter'], 
        buffer_size = option['buffer_size'])
        
    network = Network(
        n_channel = option['n_channel'], 
        n_classes = option['n_classes'], 
        image_x_size = option['image_x_size'], 
        image_y_size = option['image_y_size'],
        max_objects = option['max_objects'], 
        cell_x_size = option['cell_x_size'], 
        cell_y_size = option['cell_y_size'], 
        pool_mode = option['pool_mode'],
        box_per_cell = option['n_boxes'], 
        batch_size = option['batch_size'],
        object_scale = option['object_scale'], 
        noobject_scale = option['noobject_scale'], 
        coord_scale = option['coord_scale'], 
        class_scale = option['class_scale'],
        noobject_thresh = 0.6, 
        recall_thresh = 0.6, 
        pred_thresh = 0.5, 
        nms_thresh = 0.4,
        is_weight_decay = option['is_weight_decay'],
        weight_decay_scale = option['weight_decay'])
    
    model = Model(
        n_channel = option['n_channel'], 
        max_objects = option['max_objects'],
        image_x_size = option['image_x_size'], 
        image_y_size = option['image_y_size'], 
        cell_x_size = option['cell_x_size'], 
        cell_y_size = option['cell_y_size'],
        n_classes = option['n_classes'],
        box_per_cell = option['n_boxes'], 
        batch_size = option['batch_size'],
        buffer_size = option['buffer_size'],
        is_valid = option['is_valid'], 
        update_function = option['update_function'], 
        learning_rate = option['learning_rate'],
        is_lr_decay = option['is_lr_decay'],
        is_observe = option['is_observe']) 
    
    if method == 'train':
	

        # 训练模型
        train_image_paths_file = os.path.join(data_dir, 'datasets', option['train_data'], 'train.txt')
        test_image_paths_file = os.path.join(data_dir, 'datasets', option['test_data'], 'valid.txt')
        traineval_image_paths_file = os.path.join(data_dir, 'datasets', option['test_data'], 'traineval.txt')
        processor.init_datasets(mode='train',
            train_image_paths_file=train_image_paths_file, 
            test_image_paths_file=test_image_paths_file,
            traineval_image_paths_file=traineval_image_paths_file)
        
        # 设置数据池，processor负责生产dataset，model负责消费dataset
        producers = []
        for i in range(option['n_processes']):
            producer = mp.Process(
                target=processor.dataset_producer_based_shm, args=(i,), name='producer%d' % (i))
            producers.append(producer)
        # 在CPU中运行生产者
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for producer in producers:
            producer.start()
        # 在GPU中运行消费者
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        model.train(
            processor, network, 
            backup_dir=os.path.join(data_dir, 'backup', option['seq']), 
            logs_dir=os.path.join(data_dir, 'logs', option['seq']), 
            n_iters=option['n_iter'])
        
    elif method == 'test':
	# fetch one batch of data
        print("Start fetching data")
        test_image_paths_file = os.path.join(data_dir, 'datasets', option['train_data'], 'test.txt')
        processor.load_datasets('train', image_paths_file=test_image_paths_file)
        dataset = processor.trainsets
        batch_indexs, batch_images, batch_labels = processor.get_random_batch(dataset,batch_size=processor.batch_size)
        
        # # testing this batch of data
        # print(dataset[1].keys())
        # print(dataset[1]["label"])
        # print("batch_indexs type: ",type(batch_indexs))
        # print("batch_images type: ",type(batch_images))
        # print("batch_labels type: ",type(batch_labels))
        # print("batch_indexs element type: ",type(batch_indexs[1]))
        # print("batch_images element type: ",type(batch_images[1]))
        # print(batch_images[1])
        # print("Shape of batch_images[1]: ",batch_images[1].shape)
        # print("Shape of batch_images[2]: ",batch_images[2].shape)
        # print("Shape of batch_images[3]: ",batch_images[3].shape)
        # print("Shape of batch_images[4]: ",batch_images[4].shape)
        # print("batch_labels element type: ",type(batch_labels[1]))
        
        # print("A sample of using numpy in cv")
        # emptyimage = numpy.zeros(batch_images[1].shape,numpy.uint8)
        
        # print(dataset[1].keys())
        # print(dataset[1]["image_path"])
        
        
        # img_np = batch_images[1]
        # img1 = cv2.imread(dataset[1]["image_path"])
        # cv2.imshow("img",img_np)
        # cv2.waitKey(5000)
        


        # set up session

        # print("Start setting session")
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(
        #         gpu_options=gpu_options, allow_soft_placement=True))
        backup_path = os.path.join(data_dir,"backup\\voc-v1")
        print("finish setting session")
        # load trained model
        # print("Start loading model")
        # meta_path = os.path.join(backup_path,'model_.ckpt.meta')
        # checkpoint_path = os.path.join(backup_path,'model_.ckpt')
        # model_saver = tf.train.import_meta_graph(meta_path)
        # model_saver.restore(sess,checkpoint_path)

        print("Finish loading model")
        
        print("Before transformation,the type of batch_images is ",type(batch_images))
        print(len(batch_images))
        batch_images = processor.convert_batch_infos(batch_images)
        print("After transformation,the shape of batch_images is ",batch_images.shape)

        batch_images = numpy.array(batch_images / 255.0,dtype='float32')

        print("After transformation,the type of batch_images is ",type(batch_images))

        # batch_images = tf.convert_to_tensor(batch_images)
        # sess.run(network.conv_layer1.weight)
        # print(network.conv_layer1.weight)
        
        # feed one batch of data and run
        # tf.get_default_graph()
        # print("Type of batch_images: ",type(batch_images))


        model.train_init(network,backup_path)

        logits = model.valid_logits
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        [logits] = model.sess.run(fetches=[model.valid_logits],
        feed_dict={model.place_holders['images']: batch_images})
        
        model.sess.run(model.valid_logits)
        print("Finish")
        print(model.valid_logits)


        # 测试某一个已经训练好的模型
        #test_image_paths_files=[os.path.join(data_dir, 'datasets', option['datas'][-1], 'test_tensor.txt')]
        #processor.init_datasets(mode='test', test_image_paths_files=test_image_paths_files)
        #os.environ['CUDA_VISIBLE_DEVICES'] = ''
        #model_path = os.path.join(store_dir, 'backup', option['seq'], option['sub_dir'], option['model'])
        #model.test_model(
            #processor=processor, network=network, model_path=model_path,
            #output_dir=os.path.join(store_dir, 'logs', option['seq']))


if __name__ == '__main__':
    print('current process id: %d' % (os.getpid()))
    parser = argparse.ArgumentParser(description='parsing command parameters')
    parser.add_argument('-method')
    parser.add_argument('-gpus')
    parser.add_argument('-name')
    arg = parser.parse_args()
    method = arg.method
    gpus = arg.gpus if arg.gpus else ''
    main(method=method, gpus=gpus)
