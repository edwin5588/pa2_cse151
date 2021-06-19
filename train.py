################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from neuralnet import *
import copy
from tqdm import tqdm
from utils import batch_datatset
import matplotlib.pyplot as plt

def train(x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    return five things -
        training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
        best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []
    best_model = None
    best_loss = math.inf

    model = NeuralNetwork(config=config)

    prev_loss = 0
    stop_counter = 0

    # create a random batch for sgd
    for epoch in tqdm(range(config['epochs'])):
        indxs = np.arange(y_train.shape[0])
        np.random.shuffle(indxs)
        x_shuffled = x_train[indxs]
        y_shuffled = y_train[indxs]
        x_batch, y_batch = batch_datatset(
            x_shuffled, y_shuffled, config['batch_size'])

        batch_loss = []
        batch_acc = []

        for batch in tqdm(range(0, x_batch.shape[0])):
            output, trn_loss = model(x_batch[batch], y_batch[batch])
            model.backward()
            batch_loss.append(trn_loss)
            batch_acc.append(np.sum(np.argmax(output, axis=1) ==
                                    np.argmax(y_batch[batch], axis=1)) / y_batch[batch].shape[0])

        if epoch == 0:
            plt.plot(batch_acc)
            plt.title('Training Accuarcy - Epoch = 1')
            plt.legend(['train'])
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.show()

            plt.plot(batch_loss)
            plt.title('Training Loss - Epoch = 1')
            plt.legend(['train'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.show()


        # calc train accuracy
        train_acc.append(np.average(batch_acc))
        train_loss.append(np.average(batch_loss))

        # perform validation
        validation_loss, validation_acc = test(model, x_valid, y_valid)
        valid_loss.append(validation_loss)
        valid_acc.append(validation_acc)

        tqdm.write('Train Acc: {} Train Loss: {} \nVAL acc: {} Val Loss: {}'.format(
            np.mean(batch_acc), np.mean(batch_loss), validation_acc, validation_loss))
        # save best model
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model = copy.deepcopy(model)
            stop_counter = 0
        else:
            stop_counter += 1
            print('+++++++++++++++++++++++++++++++++++++++++++++++')

        # early stop if loss is not improving
        if config['early_stop']:
            if stop_counter == config['early_stop_epoch']:
                print('-------------  early stopping ----------------')
                break

    return train_acc, valid_acc, train_loss, valid_loss, best_model


def predict(x, thresh = 0.5):
    '''
    y is a 2d array
    '''
    for i in range(len(x)):
        x[i] = [1 if j > thresh else 0 for j in x[i]]

    return x


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and return loss and accuracy on the test set.
    """
    output, loss = model(x_test, y_test)
    accuracy = np.sum(np.argmax(output, axis=1) ==
                      np.argmax(y_test, axis=1)) / x_test.shape[0]
    return loss, accuracy
