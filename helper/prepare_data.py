import numpy as np, pandas as pd, os
from keras.utils import load_img

def prepare_data(choice):
    match choice:
        case 1:
            df = pd.read_csv(os.getcwd() + "//datasets//age_gender.csv")

            # image size originally is 48
            img_size = 48

            # First split each pixel value and convert to float, only then we can normalize values of pixels from 0 - 255 to 0 - 1:
            df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype = "float32"))
            df['pixels'] = df['pixels'].apply(lambda x: x / 255)

            x = np.array(df['pixels'].tolist())
            # x.shape[0] = 23705, those are number of entries in db file, last argument is 1 if greyscale, 3 if rgb images
            x = x.reshape(x.shape[0], img_size, img_size, 1)

            x = np.array(x)
            y_age = np.array(df['age'])
            y_ethnicity = np.array(df['ethnicity'])
            y_age = np.array(df['gender'])
            return x, y_age, y_age, y_ethnicity, img_size
        case 2:
            directory = os.getcwd() + '//datasets//UTKFace//'

            # image size originally is 200
            img_size = 200
            
            # lists for storing labels
            image_paths = []
            age_labels = []
            gender_labels = []

            for filename in os.listdir(directory):
                image_path = os.path.join(directory, filename)
                temp = filename.split('_')
                age = int(temp[0])
                gender = int(temp[1])
                image_paths.append(image_path)
                age_labels.append(age)
                gender_labels.append(gender)

            df = pd.DataFrame()
            df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels

            x = []
            for image in df['image']:
                    img = load_img(image, color_mode = "grayscale")
                    img = img.resize((img_size, img_size), 3)
                    img = np.array(img)
                    x.append(img)

            x = np.array(x)
            x = x.reshape(len(x), img_size, img_size, 1)
            x = x/255.0

            y_age = np.array(df['gender'])
            y_age = np.array(df['age'])
            if 'ethnicity' in df.columns:
                y_ethnicity = np.array(df['ethnicity'])
            else:
                y_ethnicity = np.empty(0)
            return x, y_age, y_age, y_ethnicity, img_size
        case 3:
            directory = os.getcwd() + '//datasets//Fairface//val'

            # image size originally is 224
            img_size = 224
            
            x = []

            for filename in os.listdir(directory):
                image_path = os.path.join(directory, filename)
                img = load_img(image_path, color_mode = "grayscale")
                img = img.resize((img_size, img_size), 3)
                img = np.array(img)
                x.append(img)

            x = np.array(x)
            x = x.reshape(len(x), img_size, img_size, 1)
            x = x/255.0

            df = pd.read_csv(os.getcwd() + "//datasets//Fairface//fairface_label_val.csv")
            y_age = []
            y_gender = []
            y_ethnicity = [] 

            for entry in np.array(df['age']):
                if entry == '0-2':
                    y_age.append(0)
                elif entry == '3-9':
                    y_age.append(1)
                elif entry == '10-19':
                    y_age.append(2)
                elif entry == '20-29':
                    y_age.append(3)
                elif entry == '30-39':
                    y_age.append(4)
                elif entry == '40-49':
                    y_age.append(5)
                elif entry == '50-59':
                    y_age.append(6)
                elif entry == '60-69':
                    y_age.append(7)
                elif entry == 'more than 70':
                    y_age.append(8)

            for entry in np.array(df['gender']):
                if entry == "Male":
                    y_gender.append(0)
                elif entry == "Female":
                    y_gender.append(1)

            for entry in np.array(df['race']):
                if entry == "White":
                    y_ethnicity.append(0)
                elif entry == "Black":
                    y_ethnicity.append(1)
                elif entry == "Latino_Hispanic":
                    y_ethnicity.append(2)
                elif entry == "East":
                    y_ethnicity.append(3)
                elif entry == "Southeast Asian":
                    y_ethnicity.append(4)
                elif entry == "Indian":
                    y_ethnicity.append(5)
                elif entry == "Middle Eastern":
                    y_ethnicity.append(6)

            y_age = np.array(y_age)
            y_gender = np.array(y_gender)
            y_ethnicity = np.array(y_ethnicity)

            return x, y_age, y_age, y_ethnicity, img_size
        case _:
            return 0,0,0,0,0