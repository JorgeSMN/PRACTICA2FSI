import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

x_train_data = train_x.astype('f4')  # the samples are the four first rows of data
y_train_data = one_hot(train_y.astype(int), 10)  # the labels are in the last row. Then we encode them in one hot

x_valid_data = valid_x.astype('f4')  # the samples are the four first rows of data
y_valid_data = one_hot(valid_y.astype(int), 10)  # the labels are in the last row. Then we encode them in one hot

x_test_data = test_x.astype('f4')  # the samples are the four first rows of data
y_test_data = one_hot(test_y.astype(int), 10)  # the labels are in the last row. Then we encode them in one hot


validData_y = one_hot(valid_y, 10)
testData_y = one_hot(test_y, 10)
train_y = one_hot(train_y, 10)

x = tf.placeholder("float", [None, 28*28])  # samples, imagenes 28*28
y_ = tf.placeholder("float", [None, 10])  # labels, 10 etiquetas

W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)



print "/////////////////////////////---ENTRENAMIENTO Y VALIDACION---/////////////////////////////////////////"

batch_size = 20
errorActual=100
errorAnterior=0
comp = 10
epoch = 0
error=[]

while comp > 0.02:  #si la diferencia entre los errores es menor a un 2% para
    for jj in xrange(len(x_train_data) / batch_size):
        batch_xs = x_train_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    errorAnterior = errorActual
    errorActual = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    error.append(errorActual)
    print "El error actual es", errorActual
    comp = abs(errorActual - errorAnterior)/errorAnterior
    epoch = epoch+1
    print "Epoch #:", epoch, "Error: ", errorActual
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

print "///////////////////////////////////---TEST---///////////////////////////////////////////////"
result = sess.run(y, feed_dict={x: test_x})
bien = 0
mal = 0
bienAnterior=0


for b, r in zip(testData_y, result):
    if (np.argmax(b) == np.argmax(r)):
        bien = bien + 1
    else:
        mal = mal + 1

    print (b, "-->", r)

    if(bienAnterior < bien):
        print "ACIERTA"
    else:
        print "FALLA"
    bienAnterior = bien
    malAnterior = mal

print "Aciertos totales ", bien
print "Fallos totales: ", mal
todos = bien+mal

print "Se ha acertado en el ", (float(bien) / float(todos)) * 100, "% de los casos"
print("----------------------------------------------------------------------------------")


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.plot(error)
plt.legend(['Evolucion del error'])
plt.show()  # Let's see a sample


# TODO: the neural net!!
