from cnn_models import *
from matplotlib import pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class NewEmotionDetector:
    def __init__(self):

        self.batch_size = 64
        self.input_shape = (48, 48, 1)
        self.num_classes = 7

        self.test_datagen = ImageDataGenerator(rescale=1./255)

    def prediction(self, model_path, img):

        trained_model = load_model(model_path)
        prediction_generator = self.test_datagen.flow(img, [1])

        return trained_model.predict_generator(prediction_generator, 1)

    def base_model(self, model_name, input_shape, num_classes):

        data = pd.read_csv('all/fer2013.csv')

        pixels = data['pixels'].tolist()

        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(48, 48)

            faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        emotions = pd.get_dummies(data['emotion']).values

        X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.1, random_state=42)

        if model_name == 'tiny_XCEPTION':
            model = tiny_XCEPTION(input_shape, num_classes)
        elif model_name == 'mini_XCEPTION':
            model = mini_XCEPTION(input_shape, num_classes)
        elif model_name == 'big_XCEPTION':
            model = big_XCEPTION(input_shape, num_classes)
        elif model_name == 'simple_CNN':
            model = simple_CNN(input_shape, num_classes)
        elif model_name == 'VGG_16':
            model = VGG_16(input_shape, num_classes)
        elif model_name == 'oneLayer_cnn':
            model = oneLayer_cnn(input_shape, num_classes)
        elif model_name == 'threeLayer_cnn':
            model = threeLayer_cnn(input_shape, num_classes)
        elif model_name == 'complex_cnn':
            model = complex_cnn(input_shape, num_classes)
        else:
            raise ValueError("Only 'tiny_XCEPTION', 'mini_XCEPTION', 'big_XCEPTION', 'simple_CNN', 'VGG_16', "
                             "'oneLayer_cnn', 'threeLayer_cnn' and 'complex_cnn' models available")

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

        history = model.fit(np.array(X_train), np.array(y_train),
                            batch_size=64,
                            epochs=50,
                            verbose=1,
                            validation_data=(np.array(X_val), np.array(y_val)),
                            shuffle=True,
                            callbacks=[reduce_lr, early_stopper])

        # serialize model to JSON
        model_json = model.to_json()
        with open('new_'+model_name+'.json', 'w') as json_file:
            json_file.write(model_json)

        # save model
        model.save('new_models/'+model_name+'.h5')

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('new_accuracy_'+model_name+'.png')

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('new_loss_'+model_name+'.png')


if __name__ == "__main__":
    a = NewEmotionDetector()
    # ###### for training image of size (48, 48, 3) ######
    # a.base_model(model_name='mini_XCEPTION', input_shape=(48, 48, 3), num_classes=7)

    # ###### for training image of size (48, 48, 1) ######
    a.base_model(model_name='mini_XCEPTION', input_shape=(48, 48, 1), num_classes=7)


