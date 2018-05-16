from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.utils import to_categorical
from ActivationStudy import GradientActivationStore


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

y_train_hot = to_categorical(y_train, 10)
y_test_hot = to_categorical(y_test, 10)


x_in = Input(shape=(784,))
l = Dense(1000, activation='relu')(x_in)
for i in range(2):
    l = Dense(1000, activation='relu')(l)
output = Dense(10, activation='softmax')(l)

model = Model(inputs=x_in, outputs=output)

print (model.summary())

model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.001))


cbk = GradientActivationStore(DIR='logs', num_classes=10, record_every=1, only_weights=False)

cbbk = model.fit(x=x_train[:100], y=y_train_hot[:100], batch_size=32, epochs=3, callbacks=[cbk], validation_data=(x_test, y_test_hot))