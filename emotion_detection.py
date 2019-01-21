from cnn_models import *
from matplotlib import pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


class EmotionDetector:
    def __init__(self):
        self.batch_size = 128

        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            'all/Training',
            target_size=(48, 48),
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="categorical")

        self.validation_generator = self.test_datagen.flow_from_directory(
            'all/PublicTest',
            target_size=(48, 48),
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="categorical")

    def prediction(self, model_path, img):

        trained_model = load_model(model_path)
        prediction_generator = self.test_datagen.flow(img, [1])

        return trained_model.predict_generator(prediction_generator, 1)

    def base_model(self, model_name, input_shape, num_classes):

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
        else:
            raise ValueError("Only 'tiny_XCEPTION', 'mini_XCEPTION', 'big_XCEPTION', 'simple_CNN', 'VGG_16', "
                             "'oneLayer_cnn', 'threeLayer_cnn' models available")

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

        history = model.fit_generator(
                self.train_generator,
                steps_per_epoch=2000//self.batch_size,
                epochs=50,
                validation_data=self.validation_generator,
                validation_steps=800//self.batch_size,
                callbacks=[reduce_lr])

        # serialize model to JSON
        model_json = model.to_json()
        with open(model_name+'.json', 'w') as json_file:
            json_file.write(model_json)

        # save model
        model.save('models/'+model_name+'.h5')

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('accuracy_'+model_name+'.png')

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('loss_'+model_name+'.png')


if __name__ == "__main__":
    a = EmotionDetector()
    a.base_model(model_name='VGG_16', input_shape=(48, 48, 3), num_classes=6)

