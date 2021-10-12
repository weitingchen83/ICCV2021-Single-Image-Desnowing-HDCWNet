import cv2
import numpy as np
import os
import sys
from keras.models import load_model
from argparse import ArgumentParser
import time
import tensorflow as tf
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator

#custom
import model.model 

print('import end')

def parse_args():
    parser = ArgumentParser(description='Predict')
    parser.add_argument(
        '-dataroot', '--dataroot',
        type=str, default='./testImg',
        help='root of the image, if data type is npy, set datatype as npy'
    )
    parser.add_argument(
        '-datatype', '--datatype',
        type=str, default=['jpg','tif','png'],
        help='type of the image, if == npy, will load dataroot'
    )
    parser.add_argument(
        '-predictpath', '--predictpath',
        type=str, default='./predictImg',
        help='root of the output'
    )
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=3,
        help='batch_size'
    )
    
    return  parser.parse_args()

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    if count != total:
        sys.stdout.flush()
    else:
        print()
    
    
    
def generate_data_generator(datagenerator, X,BATCHSIZE):
    genX1 = datagenerator.flow(X,batch_size = BATCHSIZE,shuffle=False)
    count = 0
    while True:
            Xi1 = genX1.next()
            
            Xi1 = Xi1/255
            yield [Xi1]
            
if __name__== '__main__':
    
    args = parse_args()

    #read test data
    selectNames = []
    if args.datatype == 'npy':
        print('Load from npy:',args.dataroot)
        data = np.load(args.dataroot)
    else:
        data=[]
        print('Read img from:' , args.dataroot)
        fnames=os.listdir(args.dataroot)
        print('Len of the file:',len(fnames))
        count = 1
        for f in fnames:
            progress(count,len(fnames),'Loading data...')
            count+=1
            if f.split('.')[-1] in args.datatype:
                tmp=cv2.imread(args.dataroot+'/'+f)
                selectNames.append(f)
                if tmp.shape[1]<tmp.shape[0]:
                    tmp=np.rot90(tmp)
                if tmp.shape[0]!=480 or tmp.shape[1]!=640:
                    tmp=cv2.resize(tmp, (640, 480), interpolation=cv2.INTER_CUBIC)
                data.append(tmp)
        data=np.array(data)
    
    print(data.shape,'data shape')
    print('Start Padding')
    for i in range(data.shape[0]):
        progress(i+1,data.shape[0],'Paddding and convert data to YCRCB...')
        data[i]=cv2.cvtColor(data[i],cv2.COLOR_BGR2YCR_CB)
    data=np.pad(data,((0,0),(16,16),(16,16),(0,0)),'constant')
    print(data.shape,'data shape')
    #data=data/255

    if not os.path.exists(args.predictpath):
        os.mkdir(args.predictpath)
    
    #BUILD COMBINE MODEL
    print('----------Build Model----------')
    model=model.model.build_DTCWT_model((512,672,3))
    model.load_weights('./modelParam/finalmodel.h5',by_name=False)

    print('LogPath:',args.predictpath)

    val_data_gen = ImageDataGenerator(featurewise_center=False,
                         featurewise_std_normalization=False)
    
    pred=model.predict_generator(generate_data_generator(val_data_gen,data,args.batch_size),steps = data.shape[0]/args.batch_size,verbose=1)
    
    print('Save Output')
    for i in range(pred.shape[0]):
        progress(i+1,pred.shape[0],'Saving output...')
        pred[i]=np.clip(pred[i],0.0,1.0)
        if args.datatype == 'npy':
            cv2.imwrite(args.predictpath+'/'+str(i)+'.jpg',cv2.cvtColor( (pred[i]*255).astype(np.uint8), cv2.COLOR_YCrCb2BGR))
        else:
            cv2.imwrite(args.predictpath+'/'+os.path.splitext(selectNames[i])[0]+'.jpg',cv2.cvtColor( (pred[i]*255).astype(np.uint8), cv2.COLOR_YCrCb2BGR))
