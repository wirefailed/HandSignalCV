import os
import glob
import numpy as np
import cv2
import h5py

hdf5_path = './dataset.hdf5'  # address to where you want to save the hdf5 file

# read address(images) and labels from Signals
def setAddressAndLabels(path: str): 
    addressPath = f'./Signals/{path}/*/*.jpg'
    # print(f"Searching for files in: {addressPath}")  # Debug statement

    addresses = glob.glob(addressPath)
    labels = [os.path.basename(os.path.dirname(addr)) for addr in addresses] 
    # os.path.dirname gets the directory and basename gets last componenet
    # print(f"Found {len(addresses)} files in {path}") # Debug statement

    return addresses, labels

# create storage to store image and labels
def createStorageAndLabel(path: str, hdf5_file, labels):
    data_shape = (0, 300, 300, 3)  # images are 300x300 RGB
    img_dtype = np.uint8 # data type of image

    tempStorage = hdf5_file.create_dataset(f'{path}_img', shape=data_shape, maxshape=(None, 300, 300, 3), dtype=img_dtype)

    byte_labels = np.array([label.encode('utf8') for label in labels], dtype='S') # encoding label proper to fit in hdf5 file format
    hdf5_file.create_dataset(f'{path}_labels', data=byte_labels) 

    return tempStorage

# looping over all the folders and store images into the storage
def loopOverStorage(data, storage) -> None:
        for j in range(len(data)):
            for i in range(len(data[j][0])):

                # print how many images are saved every 100 images
                if i % 10 == 0 and i > 1:
                    print('Train data: {}/{}'.format(i, len(data[j][0])))
            
                # cv2 load images as BGR, convert it to RGB
                addr = data[j][0][i]

                img = cv2.imread(addr)
                img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
                # save the image and calculate the mean so far
                storage[j].resize((storage[j].shape[0] + 1,) + storage[j].shape[1:])
                storage[j][-1] = img
        

def main():

    dataSignals = ['training_set', 'valid_set', 'test_set'] # different sets of data
    dataMatrix = [] 
    for dataSet in dataSignals: # assigned all these sets and infos inside data Matrix
        tempAddres, tempLabels = setAddressAndLabels(dataSet)
        tempMatrix = [tempAddres, tempLabels]
        print(f'{dataSet} size:,', len(tempAddres))
        dataMatrix.append(tempMatrix)

    # open a hdf5 file and create earrays
    with h5py.File(hdf5_path, 'w') as hdf5_file: # writing a file
        storage = []
        for idx, dataSet in enumerate(dataSignals):
            storage.append(createStorageAndLabel(dataSet, hdf5_file, dataMatrix[idx][1]))
    
        loopOverStorage(dataMatrix, storage)

        print('HDF5 Done')
        hdf5_file.close()

if __name__ == "__main__":
    main()