import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

'''
分析一下tf的训练过程
1. keras.dataset 构建数据集
    x = tf.convert_to_tensor()
    db = tf.data.from_tensor_slices((x, y))
    db.batch(batch_size).repeat(30)
2. 构建模型
    keras.Sequential() #他是不是和torch.nn.Sequential()一样啊，但是只能建立一个较小的模型
3. 损失函数
    tf.square() #不用torch.nn.Module之类的吗？
4. 度量
    keras.metrics.Accuracy()
5. 训练
    loop:
        构建梯度tape
        计算梯度
        更新梯度
'''

# 设置GPU使用方式
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # 设置GPU为增长式占用
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True) 
  except RuntimeError as e:
    # 打印异常
    print(e)

(xs, ys),_ = datasets.mnist.load_data()
print('datasets:', xs.shape, ys.shape, xs.min(), xs.max())

batch_size = 32

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(batch_size).repeat(30) #表示将数据复制多少倍，也就是说在一个epoch中一个数据可以被训练的次数 XXX 不是这样的，这个参数相当于epoch，重复30个epoch


model = Sequential([layers.Dense(256, activation='relu'), 
                     layers.Dense(128, activation='relu'),
                     layers.Dense(10)])
model.build(input_shape=(4, 28*28))
model.summary()

optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()

for step, (x,y) in enumerate(db):

    with tf.GradientTape() as tape:
        # 打平操作，[b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28*28))
        # Step1. 得到模型输出output [b, 784] => [b, 10]
        out = model(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10) #one_hot的意思就是只有一位是1，其他的都是0
        # 计算差的平方和，[b, 10]
        loss = tf.square(out-y_onehot)
        # 计算每个样本的平均误差，[b]
        loss = tf.reduce_sum(loss) / x.shape[0]


    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


    if step % 200==0:

        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()
