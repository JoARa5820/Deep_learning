# Lab 5 Logistic Regression Classifier
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# W의 shape : X(None,2)*W(2,1) + b(1) = y(1)
W = tf.Variable(tf.random_normal([2, 1]), name='weight')  # 2 : x의 개수 / 1 : 나가는 개수(y의 개수)
b = tf.Variable(tf.random_normal([1]), name='bias')  # 1 : b는 나가는 값과 같음

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
# 위의 식 = H(x) = 1/(1+exp(-WtX)) 이고, 이를 아래처럼 sigmoid로 사용하면 간단해짐
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
# 아래식은 y가 0일 때와 1일 때로 나누어 작성해도 되지만, 간편성을 위해 통합식을 사용함
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

# cost함수를 W에 대해 직접 미분해줘도 되지만, 굳이 그럴 필요없이 GradientDescentOptimizer를 사용함
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # predicted의 값은 True 또는 False로 나오는데, 이를 tf.float32로 casting하면 1과 0값으로 반환해줌
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))  # reduce_mean으로 평균낸 값을 accuracy로 도출

# Launch graph
with tf.Session() as sess:  # 세션 생성
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())  # variable들 초기화

    # loop 돌면서 학습시킴
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
# step, cost
0 1.73078
200 0.571512
400 0.507414
600 0.471824
800 0.447585
...
9200 0.159066
9400 0.15656
9600 0.154132
9800 0.151778
10000 0.149496

Hypothesis: 
# 0.5를 기준으로 0.5보다 작으면 0, 0.5보다 크면 1
[[ 0.03074029]
 [ 0.15884677]
 [ 0.30486736]
 [ 0.78138196]
 [ 0.93957496]
 [ 0.98016882]]
 
Correct (Y): 
# Hypothesis를 기준으로 binary가 다음과 같이 정의됨
[[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
 
Accuracy:
# Correct 값과 y_data를 비교한 결과
1.0
'''
