import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import time
import matplotlib.pyplot as plt


def get_preprocessed_raw_data(filename):
    try:
        df = pd.read_csv(filename)
        print("Preprocessed datafile retrieved successfully...")
        return df
    except(FileNotFoundError):
        print(
            "Error handled. File: \"{filename}\" was not found, run the preprocessing first and check spelling of the name.")
        exit()
    # And if different error we want it to raise whatever error happens..


def process_data(df_final):

    #df_final = df_final.drop(columns=['inf_ecg.csv','inf_gsr.csv','inf_ppg.csv','pixart.csv'])
    seed_train_test = 0
    seed_test_val = 0
    # random state is a seed value
    train_data = df_final.sample(frac=0.75, random_state=seed_train_test)
    test_data = df_final.drop(train_data.index)
    Y_train = train_data[['out1', 'out2', 'out3', 'out4', 'out5', 'out6']]
    X_train = train_data.drop(
        columns=['out1', 'out2', 'out3', 'out4', 'out5', 'out6'])
    y_test = test_data[['out1', 'out2', 'out3', 'out4', 'out5', 'out6']]
    x_test = test_data.drop(
        columns=['out1', 'out2', 'out3', 'out4', 'out5', 'out6'])
    X_val, X_test, Y_val, Y_test = train_test_split(
        x_test, y_test, test_size=0.4, random_state=seed_test_val)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_val = scaler.fit_transform(X_val)

    X_train = np.asarray(X_train).astype(np.float32)
    Y_train = np.asarray(Y_train).astype(int)
    X_test = np.asarray(X_test).astype(np.float32)
    Y_test = np.asarray(Y_test).astype(int)
    X_val = np.asarray(X_val).astype(np.float32)
    Y_val = np.asarray(Y_val).astype(int)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def make_model(arg_epochs, arg_batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val):
    """
    return: epochs, batch size, accuracy, confusion matrix, time to fit the model
    """

    n_features = X_train.shape[1]

    # Define model
    model = Sequential()
    model.add(Dense(40, activation=activation_func_hidden,
              kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation=activation_func_hidden,
              kernel_initializer='he_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation=activation_func_hidden,
              kernel_initializer='he_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation=activation_func_output))
    #model.add(Dense(1, activation=None))

    # Define the optimizer
    model.compile(
        optimizer=Adam(learning_rate),
        loss=MeanSquaredError(),
        metrics=['accuracy']
    )

    start_time = time.time()
    # Fit the model
    history = model.fit(X_train, Y_train, epochs=arg_epochs,
                        batch_size=arg_batch_size, verbose=1, validation_data=(X_val, Y_val))
    time_to_fit = round(time.time()-start_time, 2)

    # Testing Data
    y_pred = model.predict(X_test)

    conf = multilabel_confusion_matrix(Y_test, y_pred.round())
    acc = round(accuracy_score(Y_test, y_pred.round()), 5)

    res = arg_epochs, arg_batch_size, acc, conf, time_to_fit
    return res, history, model, conf


def run_once(epochs, batch_size, learning_rate, activation_func_1, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores):
    """
    Creates one model of given parameters
    """
    results = []
    accuracies = []
    res, history, model, conf = make_model(epochs, batch_size, learning_rate, activation_func_1,
                                     activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val)
    results.append(res)
    accuracies.append(res[2])
    sum = 0
    for acc in accuracies:
        sum += acc
    acc_avg = sum/len(accuracies)

    for res in results:
        print_results(res[0], res[1], res[2], res[3], res[4])

    print("!!!!! MODEL FITTINGS ARE FINISHED !!!!!")
    print(
        f"Average accuracy for {epochs} epochs, {batch_size} batch size and {learning_rate} LR was: {acc_avg}")

    # Adding to compare different models. res [4] is the time to fit. About the same for all so using last is fine.
    avg_scores.append([epochs, batch_size, learning_rate, acc_avg, res[4]])
    return history, model, conf


def print_results(epochs, batch_size, acc, conf, time_to_fit):
    '''
    Prints the values given from the model
    '''
    print('---------------------')
    print('Epochs:', epochs)
    print('Batch size:', batch_size)
    print('Confusion Matrix:\n', conf)
    print('Accuracy:', acc)
    print('Time to fit:', round(time_to_fit), "seconds")


def show_figs(history, name):
    # Data from the last model fitting process
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plotting of Training vs Validation accuracy evolving with epochs
    epochs = range(1, len(acc) + 1)

    fig1 = plt.figure()

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'k', label='Validation acc')
    plt.title('Training and validation accuracy '+name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    fig2 = plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'k', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.xlabel('X-axis Label')
    plt.legend()
    plt.grid()

    plt.show()

def get_accuracies(conf, acc):
    
    for i in range(0, conf.shape[0]):
        conf_mat = conf[i]
        actual_acc = (conf_mat[0,0]+conf_mat[1,1])/conf_mat.sum()
        acc.append(actual_acc)
    
    return acc
#def main():

preprocessed_data_filename = "preprocessed_raw_data.csv"
df_preprocessed = get_preprocessed_raw_data(preprocessed_data_filename)

X_train, Y_train, X_test, Y_test, X_val, Y_val = process_data(df_preprocessed)

# FITTING NEURAL NETWORK
# Activation functions for the network
activation_func_hidden = 'tanh'
activation_func_output = 'softmax'

# For saving the average scores of different types of runs.
avg_scores = []
acc = []
history, model, conf = run_once(100, 50, 1e-3, activation_func_hidden, activation_func_output,
                          X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores)
acc = get_accuracies(conf, acc)
print('Accuracies for each class:', acc)
#history2, model = run_once(200, 10, 5e-4, activation_func_hidden, activation_func_output,
#                           X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores)
#history3, model = run_once(200, 10, 1e-4, activation_func_hidden, activation_func_output,
#                           X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores)
#history4, model = run_once(200, 10, 1e-4, activation_func_hidden,
#                           "sigmoid", X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores)
#for epochs in range(10, 1000, 10):
#    history, model = run_once(epochs, 10, 1e-4, activation_func_hidden, activation_func_output,
#                              X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores)
show_figs(history, "-3")
#show_figs(history2, "5e-4")
#show_figs(history3, "1e-4 sigmoid")
#show_figs(history4, "1e-4 sigmoid")
print(f"Average accuracies for the different runs [epochs, batch size, learning rate, average accuracy, time to fit]:{avg_scores}")

#if __name__ == "__main__":

#    main()
