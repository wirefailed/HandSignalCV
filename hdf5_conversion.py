from random import shuffle
import glob
import numpy as np
import h5py
import tables
import cv2

hdf5_path = '/Users/soo/personalProjects/HandSignalCV/dataset.hdf5'  # address to where you want to save the hdf5 file

def setAddressAndLabels(path: str) -> Tuple[List[str], List[str]]: # read addresses and labels from the 'Dataset' folder
    addressPath = f'/Users/soo/personalProjects/HandSignalCV/Signals/{path}/*/*.jpg'
    address = glob.glob(addressPath)
    labels = [os.path.basename(os.path.dirname(addr)) for addr in addrs] # os.path.dirname gets the directory and basename gets last componenet
    return address, labels

def createStorageAndLabel(path: str):
    tempStorage = hdf5_file.create_earray(hdf5_file.root, f'{path}_img', img_dtype, shape=data_shape)
    hdf5_file.create_array(hdf5_file.root, f'{path}_labels', train_labels) # create the label arrays and copy the labels data in them
    return tempStorage


def loopOverStorage(data, storage) -> None:
        for j in len(data):
            for i in range(len(data[j][0])):

                # print how many images are saved every 100 images
                if i % 10 == 0 and i > 1:
                    print('Train data: {}/{}'.format(i, len(data[j][0])))
            
                # read an image and resize to (224, 224)
                # cv2 load images as BGR, convert it to RGB
                addr = data[j][0][i]
                #print(addr)
                img = cv2.imread(addr)
                img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
                # save the image and calculate the mean so far
                storage[j].append(img[None])
        

def main():

    dataSignals = ['training_set', 'valid_set', 'test_set']
    dataMatrix = []
    for dataSet in dataSignals: # assigned all these sets and infos inside data Matrix
        tempAddres, tempLabels = setAddressAndLabels(dataSignals)
        tempMatrix = [tempAddres, tempLabels]
        print(f'{dataSet} size:,' + len(tempAddres))
        dataMatrix.append(tempMatrix)


    data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
    img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved

    # check the order of data and chose proper data shape to save images
    if data_order == 'th':
        data_shape = (0, 3, 300, 300)
    elif data_order == 'tf':
        data_shape = (0, 300, 300, 3)

    # open a hdf5 file and create earrays
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    try:
        storage = []
        for dataSet in dataSignals:
            storage.append(createStorageAndLabel(dataSet))
    
        loopOverStorage(dataMatrix, storage)

        print('HDF5 Done')
    finally:
        print('In Finally')
        hdf5_file.close()