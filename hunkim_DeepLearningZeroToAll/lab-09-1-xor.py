# Lab 9 XOR
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

# -------------------------------------------------------------------- #
W = tf.Variable(tf.random_normal([2, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
# y 결과값이 0 또는 1이므로 복잡한 soft_max를 사용할 필요없이 logistic regression을 사용해줌
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)  # 단층퍼셉트론 : XOR 문제 해결 불가
# -------------------------------------------------------------------- #

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, w_val = sess.run(
                  [train, cost, W], feed_dict={X: x_data, Y: y_data}
        )
        if step % 100 == 0:
            print(step, cost_val, w_val)

    # Accuracy report
    # 여기서는 데이터가 4개밖에 없기 때문에 학습,테스트 데이터로 나누지 않고, predicted, accuracy 구할때도 그냥 학습데이터 사용함
    h, c, a = sess.run(
              [hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data}
    )
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

'''
# 결과는 좋지 않게 나옴
# 이유 : XOR 문제는 단층퍼셉트론으로는 해결할 수 없기 때문
Hypothesis:  
[[ 0.5]
 [ 0.5]
 [ 0.5]
 [ 0.5]]
 
Correct:  
[[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
Accuracy:  0.5
'''
