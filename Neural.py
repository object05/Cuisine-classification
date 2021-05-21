import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import *

import matplotlib.pyplot as plt

class NN:
    def __init__(self):
        is_trained = False
    
    # trains the model
    def train(self, X_train, Y_train, X_validation, Y_validation):
        
        # create NN architecture
        self.model = Sequential()
        self.model.add(Dense(2000, activation='relu', input_dim=X_train.shape[1]))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(Y_train.shape[1], activation='softmax'))
    
        # compile the model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
    
        # train the model
        history = self.model.fit(X_train, Y_train, epochs=25, batch_size=128, verbose=1, validation_data=(X_validation, Y_validation))
        self.is_trained = True
                
        # plot train/validation accuracy as a function of epochs
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle('Accuracy as a function of training epochs')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epochs')
        ax.plot(range(len(history.history['acc'])), history.history['acc'], color='blue')
        ax.plot(range(len(history.history['acc'])), history.history['acc'], marker='o', color='blue', label='Training accuracy')
        ax.plot(range(len(history.history['val_acc'])), history.history['val_acc'], color='#f39c12')
        ax.plot(range(len(history.history['val_acc'])), history.history['val_acc'], marker='o', color='#f39c12', label='Validation accuracy')

        ax.legend()
        plt.show()
        plt.close(fig)
    
    # returns prediction results for samples of X_test
    def predict(self, X_test):
        assert self.is_trained, 'model not trained yet!'
        
        # predict
        Y_predicted = self.model.predict(X_test)
        
        # convert one-hot predictions to cuisine labels
        predictions = []
        for i in range(Y_predicted.shape[0]):
            predictions.append(id_to_cuisine[np.argmax(Y_predicted[i, :])])
                
        return predictions
    
    # validates already trained model against X_test, Y_test
    def validate(self, X_test, Y_test):
        assert self.is_trained, 'model not trained yet!'
        predictions = self.predict(X_test)
        ground_truth_labels = [ id_to_cuisine[np.argmax(Y_test[i, :])] for i in range(Y_test.shape[0])]
        common_validate(ground_truth_labels, predictions)
    
    # writes out prediction of already trained model on samples of X_test
    def write_prediction(self, infile, outfile):
        assert self.is_trained, 'model not trained yet!'
        
        X = create_supervised_one_hot(infile, create_y=False)
        ids = []
        with open(infile, 'r') as file:
            data = json.load(file)
            for recipe in data:
                ids.append(recipe['id'])
        
        with open(outfile, 'w') as out_file:
            out_file.write('id,cuisine\n')
            predictions = self.predict(X)
            assert len(ids) == len(predictions), 'these should be equal'
            N = len(ids)
            for i in range(N):
                out_file.write('{},{}\n'.format(ids[i], predictions[i]))