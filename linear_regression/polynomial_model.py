import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# generate training data
x_train = np.linspace(-1, 1, 101)

num_coeffs = 6
y_train_coeffs = [1, 2, 3, 4, 5, 6]
y_train = 0
for i in range(num_coeffs):
    y_train += y_train_coeffs[i] * np.power(x_train, i)

y_train += np.random.randn(*x_train.shape) * 1.5

learning_rate = 0.01
training_epochs = 40

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Define our polynomial model
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)

# Set up the parameter vector to all zero
w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)

lost = tf.reduce_sum(tf.square(Y-y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(lost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    for (x, y) in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X: x, Y: y})

w_val = sess.run(w)

sess.close()

plt.scatter(x_train, y_train)

predict_y = 0
for i in range(num_coeffs):
    predict_y += w_val[i] * np.power(x_train, i)

plt.plot(x_train, predict_y, 'r')
plt.show()