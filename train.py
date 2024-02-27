进口 张量流作为tf公司
进口 困难
从 张量流硬层进口Conv1D、Dense、BatchNormalization、LayerNormalizion、Activation、GlobalAveragePooling1D、ReLU、Reshape、MaxPool1D、AverageCooling1D
从 张量流.硬模型进口模型
从 张量流硬正则化器进口第二语言
从 很难进口输入
进口numpy公司作为净现值
从 硬质调节器进口第二语言
进口 Matplotlib.打印作为平板电脑
从 困难进口层
从 科学.io进口 负载垫
从 困难进口后端作为K（K）
从 硬质层.芯进口兰姆达

batch_size = sixteen
epochs = one hundred
num_classes = seven
length = one thousand and twenty-four
BatchNorm =真的
number = one hundred
normal =真的
rate = [ zero point eight , zero point two ]

data1 = 负载垫 ( &#39;&#39; )
label = 负载垫 ( &#39;&#39; )

x_train1 = data1 [ &#39;&#39; ]
x_test1 = data1 [ &#39;&#39; ]
y_train1 = label [ &#39;&#39; ]
y_test1 = label [ &#39;&#39; ]

x_train1, x_test1 = x_train1 [：，：，np 新轴 ]，x_测试1 [：，：，np 新轴 ]

 定义abs_后端 (输入 ) :
返回英国防抱死制动系统 (输入 )

 定义expand_dim_后端 (输入 ) :
返回英国展开dims (输入， one )

 定义sign_backend（签名后端） (输入 ) :
返回英国签名 (输入 )

 定义 RSG公司 (模型、nb块、出通道 ) :
# conv 1
conv_1 = 卷积1D (频道外， three, strides= one, padding=“相同”,kernel_regularizer=第二语言 (1页-4页 ), kernel_initializer=&#39;正常（_N）&#39; ) (模型 )
x1 =批次规格化 ( ) (conv_1 )
x1 =激活 ( “时钟” ) (x1个 )
a1 = 卷积1D (频道外， three, strides= one, padding=“相同”,kernel_regularizer=第二语言 (1页-4页 ), kernel_initializer=&#39;正常（_N）&#39; ) (x1个 )
a1 = 层规范化 ( ) (a1级 )
a1 =激活 ( “时钟” ) (a1级 )
#GC模块1
context = 卷积1D ( one , one, strides= one, padding=“相同”,kernel_regularizer=第二语言 (1页-4页 ), kernel_initializer=&#39;正常（_N）&#39; ) (a1级 )
context =激活 (“SoftMax” ) (上下文 )
context = tf.乘 (a1，上下文 )
#计算B T
x2 =稠密 ( one // sixteen ) (上下文 )
x2 =批次规格化 ( ) (2个 )
x2 =激活 ( “时钟” ) (2个 )
x2 =稠密 ( one ) (2个 )
#计算全局平均值
x_abs =兰姆达 (abs_后端 ) (2个 )
abs_mean = GlobalAveragePooling1D公司 ( ) (x_磅 )
x =稠密 ( one ) (绝对值（_M） )
x =批次规格化 ( ) (x个 )
x =激活 ( “时钟” ) (x个 )
x =稠密 ( one ) (x个 )
x =激活 (“乙状结肠” ) (x个 )
scales =兰姆达 (expand_dim_后端 ) (x个 )
#计算阈值
    thres = keras.layers.multiply([abs_mean, x])
    # Soft thresholding
    sub = keras.layers.subtract([x_abs, thres])
    zeros = keras.layers.subtract([sub, sub])
    n_sub = keras.layers.maximum([sub, zeros])
    residual = keras.layers.multiply([Lambda(sign_backend)(x), n_sub])

    model = layers.add([x1, residual])
    return model

def RSG1(model, nb_blocks, out_channels):
     # conv 1 
    conv_1 = Conv1D(out_channels, 3, strides=2, padding='same',kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(model)
    x1 = BatchNormalization()(conv_1) 
    x1 = Activation('relu')(x1)  
    a1 = Conv1D(out_channels, 3, strides=1, padding='same',kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(x1)
    a1 = LayerNormalization()(a1) 
    a1 = Activation('relu')(a1)
    # GC module 1
    context = Conv1D(1, 1, strides=1, padding='same',kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(a1)
    context = Activation('softmax')(context)
    context = tf.multiply(a1, context)
    # Calculate B T
    x2 = Dense(1//16)(context)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)    
    x2 = Dense(1)(x2)
    # Calculate global means
    x_abs = Lambda(abs_backend)(x2)
    abs_mean = GlobalAveragePooling1D()(x_abs)
    x = Dense(1)(abs_mean)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)  
    scales = Lambda(expand_dim_backend)(x)  
    # Calculate thresholds
    thres = keras.layers.multiply([abs_mean, x])
    # Soft thresholding
    sub = keras.layers.subtract([x_abs, thres])
    zeros = keras.layers.subtract([sub, sub])
    n_sub = keras.layers.maximum([sub, zeros])
    residual = keras.layers.multiply([Lambda(sign_backend)(x), n_sub])

    model = layers.add([x1, residual])
    return model

input_1 = Input(shape=(1024, 1))

# feature extraction
model1 = Conv1D(4, 3, 4,padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_1)
print(model1.shape)
model1 = RSG(model1, 1, 4)
print(model1.shape)
model1 = RSG1(model1,1,8)
print(model1.shape)
model1 = RSG1(model1,1,8)
print(model1.shape)
model1 = RSG1(model1,1,16)
print(model1.shape)
model1 = RSG1(model1,1,16)
print(model1.shape)
model1 = BatchNormalization()(model1)
model1 = Activation('relu')(model1)
flatten1 = layers.GlobalAveragePooling1D(name="GAP")(model1)   

output = layers.Dense(units = num_classes, activation='softmax')(flatten1)
model = Model(inputs=input_1 ,outputs=output)
#Adam
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history4 =model.fit(x_train1, y_train1, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test1, y_test1))

hidden_feature_result = Model(inputs=model.input,outputs=model.get_layer("GAP").output)
TSNE_result = hidden_feature_result.predict(x_test1)

score = model.evaluate(x=x_test1, y=y_test1, verbose=0)
print("测试集上的损失：", score[0])
打印 ( &#34;Accuracy on test set:&#34;, score [1] )
