import glob
import numpy as np
import cv2
from numpy.lib.npyio import load
from skimage.feature import hog
from skimage.transform import rescale
from sklearn import preprocessing
import pickle

class Dataset:

    scalar_file_name = 'scalar'

    # Min Number of Files in Each Class = 1000

    def __init__(self, recaptcha_data_location, batch_size=32, max_each_class=1000, train_percent = 0.8, resolution = (1920, 1080)):
        self.recaptcha_data_location = recaptcha_data_location

        # Initialize 
        self.batch_size = batch_size
        self.current_index = 0
        self.max_each_class = max_each_class
        self.train_percent = train_percent
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Initialize x data locations and y labels
        self.x_data_locations = []
        self.y_labels = []
        test_x_locations = []
        test_y_labels = []

        # Red Threshold
        self.red_threshold_bottom = 100
        self.red_threshold_top = 200


        # Labels
        self.label_dict = {
                'clicked': 0,
                'not_clicked': 1,
        }

        # Set Number Classes
        self.num_classes_list = [0, 1]

        # Set Resize Size -> 1920x1080 -> divided each dimension by 6.4 (300, 168.75)
        self.resize_size = (300, 169)

        # Load data
        self.load_data()
        self.shuffle_data()
        # Save out Test Data
        self.test_x_locations = self.x_data_locations[int(len(self.x_data_locations)*self.train_percent):]
        self.test_y_labels = self.y_labels[int(len(self.y_labels)*self.train_percent):]

        # Remove test data from training data
        self.x_data_locations = self.x_data_locations[:int(len(self.x_data_locations)*self.train_percent)]
        self.y_labels = self.y_labels[:int(len(self.y_labels)*self.train_percent)]
        # Calculate Scalar of Train Data
        # self.calculate_scalar()

        print(np.array(self.x_data_locations).shape)
        print(np.array(self.y_labels).shape)

        print(self.y_labels)


    def __len__(self):
        if len(self.x_data_locations) != len(self.y_labels):
            raise Exception('x_data_loaded and y_labels are not the same length')
        return len(self.x_data_locations)

    # Loads Labels and X Data Locations
    def load_data(self):
        # Glob folders in train folder
        folders = glob.glob(self.recaptcha_data_location + '\\*')
        
        for folder in folders:
            # Get label
            label = folder.split('\\')[-1]
            label_id = self.label_dict[label]

            # Get image files
            image_files = glob.glob(folder + '\\*.png')

            # Add data to x_data and y_labels
            added = 0
            for image_file in image_files:
                self.x_data_locations.append(image_file)
                self.y_labels.append(label_id)
                added += 1
                if added>=self.max_each_class:
                    break

    # Shuffles data locations and labels together
    def shuffle_data(self):
        # Shuffle data
        # Zip x locations and y labels together and turn into a list
        combined = list(zip(self.x_data_locations, self.y_labels))
        # Shuffle this list
        np.random.shuffle(combined)
        # Unzip the list after zipping it again
        self.x_data_locations, self.y_labels = zip(*combined)

    # Returns an image and label (after loading into memory)
    def get_index(self, index):
        # Get image and label at index
        image = cv2.imread(self.x_data_locations[index])
        
        # # Testing Transforms
        # (b, g, image) = cv2.split(image)
        image = image[round(self.y_res*2/5):round(self.y_res*3/5), round(self.x_res*2/5):round(self.x_res*3/5)]

        # cv2.imshow('image', image)
        # cv2.waitKey(0)

        # Reshape Image
        image = image.reshape(1, -1)
        # Normalize Image
        # image = self.scalar.transform(image)     

        # Flatten image to get rid of arbitrary first dimension after we transform the image
        image = image.flatten() 
        label = self.y_labels[index]

        return image, label

    # Calculatee Scalar for data
    def calculate_scalar(self):
        # Load all x data into ram
        x_data = []
        print('Loading x data into ram')
        loaded_counter = 0
        for x_location in self.x_data_locations:
            image = cv2.imread(x_location)

            # Testing Transforms
            image = cv2.resize(image, self.resize_size)

            x_data.append(image.flatten())
            loaded_counter += 1
            if loaded_counter % 1000 == 0:
                print('Loaded ' + str(loaded_counter) + ' images')
        print('Calculating Scalar')
        # Calculate Scalar
        self.scalar = preprocessing.StandardScaler().fit(x_data)

    def get_next_batch(self):
        # Get Batch
        x_data = []
        y_labels = []
        for i in range(self.current_index, self.current_index + self.batch_size):
            # Get image and label
            if i < len(self.x_data_locations):
                image, label = self.get_index(i)
                x_data.append(image)
                y_labels.append(label)
        # Update current index
        self.current_index += self.batch_size

        return x_data, y_labels

    def get_test_data(self):
        x_data = []
        y_labels = []
        # Loads Images
        for i in range(len(self.test_x_locations)):
            # Load Image
            image = cv2.imread(self.test_x_locations[i])

            # Testing Transforms
            # (b, g, image) = cv2.split(image)
            image = image[round(self.y_res*2/5):round(self.y_res*3/5), round(self.x_res*2/5):round(self.x_res*3/5)]

            # Reshape Image
            image = image.reshape(1, -1)
            # Normalize Image
            # image = self.scalar.transform(image)

            # Flatten Image
            image = image.flatten()
            # Get Label
            label = self.test_y_labels[i]
            # Add to list
            x_data.append(image)
            y_labels.append(label)
        return x_data, y_labels

    # Resets Index
    def reset_index(self):
        self.current_index = 0
        # Shuffle Data after each epoch
        self.shuffle_data()
