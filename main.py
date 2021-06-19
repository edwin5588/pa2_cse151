################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################
from utils import *
from train import *
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data
    x_train, y_train, x_test, y_test = load_data()
    print('Data Shape:', x_train.shape,
          y_train.shape, x_test.shape, y_test.shape)
    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)

    # Create validation set out of training data.
    ind = np.arange(len(x_train))
    np.random.shuffle(ind)
    eighty = int(np.ceil(len(x_train) * 0.8))

    x_train, y_train = x_train[ind], y_train[ind]

    x_train, y_train, x_val, y_val = x_train[0:eighty], y_train[0:eighty], x_train[eighty:], y_train[eighty:]

    # Any pre-processing on the datasets goes here.
    # vectorize the datasets
    x_train = np.array([img_2_vec(a) for a in x_train])
    x_val = np.array([img_2_vec(a) for a in x_val])
    x_test = np.array([img_2_vec(a) for a in x_test])

    img = np.reshape(x_train[5], (32, 32))
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    print(y_train[5])
    # normalize everything
    x_train = normalize_z(x_train)
    x_val = normalize_z(x_val)
    x_test = normalize_z(x_test)

    print('Preprocess data shape: ', x_train.shape, y_train.shape, x_val.shape,
          y_val.shape, x_test.shape, y_test.shape)

    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.title('Training vs. Validation Accuarcy')
    plt.legend(['train', 'val'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Training vs. Validation Loss')
    plt.legend(['train', 'val'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)
