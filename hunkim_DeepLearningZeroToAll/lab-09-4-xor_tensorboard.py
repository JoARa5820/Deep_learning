# Lab 9 XOR

# -------------------------------------------------------------- #
# 1. From TF graph, decide which tensors you want to log
# w2_hist = tf.summary.histogram('weights2', w2)
# cost_summ = tf.summary.scalar('cost', cost)

# 2. Merge all summaries
# summary = tf.summary.merge.all()

# 3. Create writer and add graph
# # Create summary writer
# writer = tf.summary.FileWriter('./logs')
# writer.add_graph(sess.gragh)

# 4. Run summary merge and add_summary
# s, _ = sess.run([summary, optimizer], feed_dict = feed_dict)
# writer.add_summary(s, global_step = global_step)

# 5. Launch TensorBoard
# $ tensorboard --logdir=./logs
# -------------------------------------------------------------- #

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2], name="x")
Y = tf.placeholder(tf.float32, [None, 1], name="y")

# name_scope : 그래프 그릴 때 계층(layer)별로 묶어서 정리해주기 때문에 보기 편해짐
with tf.name_scope("Layer1"):
    W1 = tf.Variable(tf.random_normal([2, 2]), name="weight_1")
    b1 = tf.Variable(tf.random_normal([2]), name="bias_1")
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)


with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([2, 1]), name="weight_2")
    b2 = tf.Variable(tf.random_normal([1]), name="bias_2")
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Hypothesis", hypothesis)

# cost/loss function
with tf.name_scope("Cost"):
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    tf.summary.scalar("Cost", cost)

with tf.name_scope("Train"):
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
tf.summary.scalar("accuracy", accuracy)  # 결과값이 여러개일 경우, tf.summary.scalar 대신 tf.summary.histogram 이용해줌

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")  # 어느 위치에 파일 저장할지 입력 : ./위치
    writer.add_graph(sess.graph)  # Show the graph

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, summary, cost_val = sess.run(
            [train, merged_summary, cost], feed_dict={X: x_data, Y: y_data}
        )
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step, cost_val)

    # Accuracy report
    h, p, a = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data}
    )
    
    print(f"\nHypothesis:\n{h} \nPredicted:\n{p} \nAccuracy:\n{a}")

# Launch TensorBoard :
# 터미널에서 'tensorboard --logdir=./위치' 쳐주면 그래프 볼 수 있음
# => 폴더 속 파일명까지 적어주면 해당 파일의 그래프만 볼 수 있지만,
#    폴더명까지만 적어주면 해당 폴더 속 여러 그래프를 비교하며 볼 수 있음
#    (ex. learning_rate에 따른 cost값들 비교)
# local에서 사용할 때 : ssh -L local_port:127.0.0.1:remote port username@server.com
# username@server.com : remote server / remote port : remote server의 포트번호로 정해져있음

"""
Hypothesis:
[[6.1310326e-05]
 [9.9993694e-01]
 [9.9995077e-01]
 [5.9751470e-05]] 
Predicted:
[[0.]
 [1.]
 [1.]
 [0.]] 
Accuracy:
1.0
"""
