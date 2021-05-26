#-*- codeing = utf-8 -*-
#@Time :2021/5/25 12:34
#@Author :Onion
#@File :train.py
#@Software :PyCharm
import tensorflow.keras as keras
from preprocess import generate_training_sequences,SEQUENCE_LENGTH
# 定义神经网络的参数
# 1、定义输出层的神经元的个数,与mapping.json中的映射数相同
OUTPUT_UNITS = 38
# 2、损失函数使用交叉熵损失函数
LOSS = "sparse_categorical_crossentropy"
# 3、学习率
LEARNING_RATE = 0.001
# 4、训练次数
EPOCHS = 50
# 5、每一批的大小batch_size = SEQUENCE_LENGTH
BATCH_SIZE = 64
# 6、定义神经元的数量（由于这里只使用了单层的LSTM隐藏层,因此在这个单层的神经网络又256个神经元）
NUM_UNITS = [256]
# 多层的表示方式
# NUM_UNITS = [256,256,256] # 这样就可以表示使用了三层LSTM隐藏层
# 7、模型保存路径
SAVE_MODEL_PATH = "model.h5"

# 一:模型构建函数
def build_model(output_units, num_units, loss, learning_rate):
    # 构建网络结构
    # 1、输入层
    input_layer = keras.layers.Input(shape=(None, output_units))

    # 2、隐藏层,使用了Dropout的方法来防止模型过拟合
    x = keras.layers.LSTM(num_units[0])(input_layer)
    x = keras.layers.Dropout(0.2)(x)

    # 3、输出层,使用sotfmax进行映射,将结果映射到0~1,映射个个数为output_units
    output_layer = keras.layers.Dense(output_units, activation="softmax")(x)

    # 4、定义模型,将input_layer和output_layer当作参数传入即可
    model = keras.Model(input_layer,output_layer)

    # 整合模型
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    model.summary()

    return model
# 二:定义训练函数
def train(output_units = OUTPUT_UNITS,num_units = NUM_UNITS,loss = LOSS,learning_rate = LEARNING_RATE):
    #   步骤
    #   1、生成训练序列
    inputs,targets = generate_training_sequences(SEQUENCE_LENGTH)

    #   2、搭建神经网络
    model = build_model(output_units,num_units,loss,learning_rate)
    #   3、训练模型
    model.fit(inputs,targets,epochs=EPOCHS,batch_size=BATCH_SIZE)
    #   4、保存模型
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()