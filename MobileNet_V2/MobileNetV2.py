from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras import Input

def expansion_block(x, t, filters, block_id):
    prefix = 'block_{}_'.format(block_id)
    total_filters = t * filters
    x = Conv2D(total_filters, (1, 1), padding='same', use_bias=False, name=prefix + 'expand')(x)
    x = BatchNormalization(name=prefix + 'expand_bn')(x)
    x = ReLU(6, name=prefix + 'expand_relu')(x)
    return x

def depthwise_block(x, stride, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = DepthwiseConv2D((3, 3), strides=(stride, stride), padding='same', use_bias=False, name=prefix + 'dw_conv')(x)
    x = BatchNormalization(name=prefix + 'dw_bn')(x)
    x = ReLU(6, name=prefix + 'dw_relu')(x)
    return x

def projection_block(x, out_channels, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = Conv2D(out_channels, (1, 1), padding='same', use_bias=False, name=prefix + 'compress')(x)
    x = BatchNormalization(name=prefix + 'compress_bn')(x)
    return x

def bottle_neck(x, t, filters, out_channels, stride, block_id):
    y = expansion_block(x, t, filters, block_id)
    y = depthwise_block(y, stride, block_id)
    y = projection_block(y, out_channels, block_id)
    if y.shape[-1] == x.shape[-1]:
        y = Add()([x, y])
    return y

def MobileNetV2(input_shape=(224, 224, 3), n_classes=1000):
    input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(input)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(6, name='Conv1_relu')(x)
    x = depthwise_block(x, stride=1, block_id=1)
    x = projection_block(x, out_channels=16, block_id=1)

    x = bottle_neck(x, 6, x.shape[-1], out_channels=24, stride=2, block_id=2)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=24, stride=1, block_id=3)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=32, stride=2, block_id=4)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=32, stride=1, block_id=5)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=32, stride=1, block_id=6)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=64, stride=2, block_id=7)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=64, stride=1, block_id=8)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=64, stride=1, block_id=9)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=64, stride=1, block_id=10)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=96, stride=1, block_id=11)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=96, stride=1, block_id=12)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=96, stride=1, block_id=13)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=160, stride=2, block_id=14)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=160, stride=1, block_id=15)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=160, stride=1, block_id=16)
    x = bottle_neck(x, 6, x.shape[-1], out_channels=320, stride=1, block_id=17)

    x = Conv2D(1280, (1, 1), padding='same', use_bias=False, name='last_conv')(x)
    x = BatchNormalization(name='last_bn')(x)
    x = ReLU(6, name='last_relu')(x)

    x = GlobalAveragePooling2D(name='global_average_pool')(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model

model = MobileNetV2()
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()





