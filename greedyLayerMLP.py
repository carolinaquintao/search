# linear algebra
import numpy as np
# Algorithms
import scipy.io
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from matplotlib import pyplot

# # Helper: Early stopping.
# tensorboard = TensorBoard(log_dir='./logs', batch_size=512)
# reduceLROnPlateau = ReduceLROnPlateau(monitor='mean_squared_error')
# early_stopper = EarlyStopping(patience=20)
# checkpoint = ModelCheckpoint("./checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=0, save_best_only=True, mode='max')
#
# K.tensorflow_backend._get_available_gpus()
#
# # Helper: Early stopping.
# tensorboard = TensorBoard(log_dir='./logs', batch_size=512)
# reduceLROnPlateau = ReduceLROnPlateau(monitor='val_acc')
# early_stopper = EarlyStopping(patience=20)
# checkpoint = ModelCheckpoint("./checkpoint/FT-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc',
#                              verbose=0, save_best_only=True, mode='max')
from keras.utils import to_categorical
# ValueError: You are passing a target array of shape (7200, 1) while using as loss `categorical_crossentropy`. `categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, classes).
# Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.

mat = scipy.io.loadmat(
    'Y:/AnaSiravenha/BCI/Artigo_Ext_KDMILE/Testes_Artigo_IJCNN/Matrizes/X_train_test_Y_train_test_China_S11.mat',
    squeeze_me=True, struct_as_record=False)

trainX = np.transpose(mat['X_train'])
trainy = to_categorical(mat['Y_train'])

X_test = np.transpose(mat['X_test'])
Y_test = mat['Y_test']

newArray = np.random.randint(0, 75000, size=(1, 4000))
X_test = X_test[newArray[0]]
Y_test = to_categorical(Y_test[newArray[0]])


# define and fit the base model
def get_base_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(10, input_dim=57, activation='relu'))#, kernel_initializer='he_uniform'
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(3, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.7)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=100, verbose=0)
    return model


# evaluate a fit model
def evaluate_model(model, trainX, testX, trainy, testy):
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    return train_acc, test_acc


# add one new layer and re-train only the new layer
def add_layer(model, trainX, trainy):
    # remember the current output layer
    output_layer = model.layers[-1]
    # remove the output layer
    model.pop()
    # mark all remaining layers as non-trainable
    for layer in model.layers:
        layer.trainable = False
    # add a new hidden layer
    model.add(Dense(10, activation='sigmoid'))#, kernel_initializer='he_uniform'
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    # re-add the output layer
    model.add(output_layer)
    # fit model
    model.fit(trainX, trainy, epochs=100, verbose=0)


# get the base model
model = get_base_model(trainX, trainy)
# evaluate the base model
scores = dict()
train_acc, test_acc = evaluate_model(model, trainX, X_test, trainy, Y_test)
print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
scores[len(model.layers)] = (train_acc, test_acc)
# add layers and evaluate the updated model
n_layers = 20
for i in range(n_layers):
    # add layer
    add_layer(model, trainX, trainy)
    # evaluate model
    train_acc, test_acc = evaluate_model(model, trainX, X_test, trainy, Y_test)
    print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
    # store scores for plotting
    scores[len(model.layers)] = (train_acc, test_acc)
# plot number of added layers vs accuracy
pyplot.plot(scores.keys(), [scores[k][0] for k in scores.keys()], label='train', marker='.')
pyplot.plot(scores.keys(), [scores[k][1] for k in scores.keys()], label='test', marker='.')
pyplot.legend()
pyplot.show()





