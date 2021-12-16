# Lab 6 Softmax Classifier
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

# y_data : 레이블이 여러개(A/B/C)이기 때문에 원-핫 인코딩으로 표현해줌
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# matmul : 행렬곱
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
# Y * tf.log(hypothesis)를 합한 후(reduce_sum) 평균 구해줌(reduce_mean)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

            if step % 200 == 0:
                print(step, cost_val)

    print('--------------')
    # Testing & One-hot encoding
    # 학습이 끝난 후 hypothesis에 X값을 넣어줌(feed_dict) : soft_max를 통과하면서 확률 합을 1이 되게 만들어줌 -> 가장 큰 값이 정답임을 표시해주는 argmax를 통과시켜줌
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    # X(feed_dict)를 한번에 합쳐서 입력시켜줄수도 있음
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))

'''
0 6.926112
200 0.6005015
400 0.47295815
600 0.37342924
800 0.28018373
1000 0.23280522
1200 0.21065344
1400 0.19229904
1600 0.17682323
1800 0.16359556
2000 0.15216158
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]] [1]  # [1]이 정답
-------------
[[0.9311919  0.06290216 0.00590591]] [0]  # [0]이 정답
-------------
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]  # [2]이 정답
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]
 [9.3119192e-01 6.2902197e-02 5.9059085e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]] [1 0 2]  # [1, 0, 2]이 정답
'''
