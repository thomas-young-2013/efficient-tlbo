from __future__ import division, print_function, absolute_import

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping

"""hyperparameters: pooling_type, lr, momentum, decay, nesterov, batch_size, dropout, regularizer, lr_reductions
"""

from sklearn.model_selection import StratifiedShuffleSplit


def balanced_split(X, Y, proportion=0.1, random_state=42):
    X_train, X_valid, Y_train, Y_valid = None, None, None, None
    train_num, val_num = 0, 0
    sss = StratifiedShuffleSplit(n_splits=1, test_size=proportion, random_state=random_state)
    for train_index, test_index in sss.split(X, Y):
        train_num, val_num = len(train_index), len(test_index)
        print("Using {} for training and {} for validation".format(train_num, val_num))
        X_train, X_valid = X[train_index], X[test_index]
        Y_train, Y_valid = Y[train_index], Y[test_index]
    return X_train, X_valid, Y_train, Y_valid, train_num, val_num


def create_model(img_size, cls_num, pooling, lr, decay, nesterov, momentum, regularizer, dropout):
    base_model = ResNet50(include_top=False, weights=None,
                                   input_shape=(img_size, img_size, 3), pooling=pooling)
    x = base_model.output
    x = Dropout(dropout)(x)
    output = Dense(cls_num, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=output)
    for layer in model.layers:
        layer.W_regularizer = l2(regularizer)
        layer.trainable = True

    opt = SGD(lr=lr, momentum=momentum, nesterov=nesterov, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def train_model(cls_num, img_size, epoch_num, params, train, proportion, seed, is_categorical=False):
    pooling = params['pooling']
    lr = params['lr']
    decay = params['decay']
    nesterov = params['nesterov']
    momentum = params['momentum']
    regularizer = params['regularizer']
    dropout = params['dropout']
    lr_reduction = params['lr_redcution']
    batch_size = params['batch_size']

    x, y = train
    if not is_categorical:
        train[1] = to_categorical(y, cls_num)

    gen = ImageDataGenerator(
        rotation_range=40.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )

    x_train, x_val, y_train, y_val, train_num, val_num = balanced_split(x, y, proportion=proportion, random_state=seed)
    x_val = x_val/255.
    model = create_model(img_size, cls_num, pooling, lr, decay, nesterov, momentum, regularizer, dropout)

    def lr_schedule(epoch):
        learning_rate = lr
        if epoch < epoch_num*0.5:
            pass
        elif epoch < epoch_num*0.75:
            learning_rate = lr*lr_reduction
        else:
            learning_rate = lr*lr_reduction*lr_reduction
        print('learning rate:', learning_rate)
        return learning_rate

    learning_rate_reduction = LearningRateScheduler(lr_schedule)

    model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size,
                        epochs=epoch_num,
                        verbose=1,
                        shuffle=True,
                        validation_data=(x_val, y_val),
                        callbacks=[learning_rate_reduction])

    score = model.evaluate(x_val, y_val)
    print('Accuracy on Validation Set', score[1])
    return score[1]


if __name__ == "__main__":
    cls_num = 10
    img_size = 32
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, cls_num)
    y_test = to_categorical(y_test, cls_num)
    params = {
        "lr": 0.001,
        "decay": 0.0001,
        "momentum": 0.9,
        "nesterov": True,
        "regularizer": 1e-4,
        "dropout": 0.3,
        "lr_redcution": 0.2,
        "batch_size": 32,
        "pooling": "avg"
    }
    acc = train_model(cls_num, img_size, epoch_num=50, params=params,
                train=(x_train, y_train), proportion=0.2, seed=32, is_categorical=True)
    print('the accuracy is', acc)
