import os
import glob
import numpy as np
import cv2
import h5py

hdf5_path = '/Users/soo/personalProjects/HandSignalCV/dataset.hdf5'  # address to where you want to save the hdf5 file

def setAddressAndLabels(path: str): # read addresses and labels from the 'Dataset' folder
    addressPath = f'/Users/soo/personalProjects/HandSignalCV/Signals/{path}/*/*.jpg'
    address = glob.glob(addressPath)
    labels = [os.path.basename(os.path.dirname(addr)) for addr in address] # os.path.dirname gets the directory and basename gets last componenet
    return address, labels

def createStorageAndLabel(path: str, hdf5_file):
    tempStorage = hdf5_file.create_dataset(f'{path}_img', img_dtype, shape=data_shape)
    hdf5_file.create_dataset(hdf5_file.root, f'{path}_labels', train_labels) # create the label arrays and copy the labels data in them
    return tempStorage


def loopOverStorage(data, storage) -> None:
        for j in range(len(data)):
            for i in range(len(data[j][0])):

                # print how many images are saved every 100 images
                if i % 10 == 0 and i > 1:
                    print('Train data: {}/{}'.format(i, len(data[j][0])))
            
                # cv2 load images as BGR, convert it to RGB
                addr = data[j][0][i]
                #print(addr)
                img = cv2.imread(addr)
                img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
                # save the image and calculate the mean so far
                storage[j].resize((storage[j].shape[0] + 1,) + storage[j].shape[1:])
                storage[j][-1] = img
        

def main():

    dataSignals = ['training_set', 'valid_set', 'test_set']
    dataMatrix = []
    for dataSet in dataSignals: # assigned all these sets and infos inside data Matrix
        tempAddres, tempLabels = setAddressAndLabels(dataSignals)
        tempMatrix = [tempAddres, tempLabels]
        print(f'{dataSet} size:,', len(tempAddres))
        dataMatrix.append(tempMatrix)

    # open a hdf5 file and create earrays
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        storage = []
        for dataSet in dataSignals:
            storage.append(createStorageAndLabel(dataSet, hdf5_file))
    
        loopOverStorage(dataMatrix, storage)

        print('HDF5 Done')
        hdf5_file.close()

if __name__ == "__main__":
    main()