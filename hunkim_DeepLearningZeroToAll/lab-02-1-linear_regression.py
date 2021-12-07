# Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# X and Y data
## train 값을 여기서 선언하지 않고 아래 37행에서
## _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {X:[1,2,3],Y:[2,3,4]})로도 표현 가능
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let TensorFlow figure it out
## W과 b는 Variable로 선언
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Our hypothesis XW+b
## 가설(식) 선언
hypothesis = x_train * W + b

# cost/loss function
## cost 함수 선언 : 평균제곱오차 / square : 제곱
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    ## tf.Variable(W,b) 실행시키기 전에 tf,global_variables_initializer() 실행시켜줘야함
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])  # train값은 print 하지않을 것이라서 _로 놓음

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

# Learns best fit W:[ 1.],  b:[ 0.]
"""
0 2.82329 [ 2.12867713] [-0.85235667]
20 0.190351 [ 1.53392804] [-1.05059612]
40 0.151357 [ 1.45725465] [-1.02391243]
...
1960 1.46397e-05 [ 1.004444] [-0.01010205]
1980 1.32962e-05 [ 1.00423515] [-0.00962736]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
"""
