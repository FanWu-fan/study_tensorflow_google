# Chapter6 图像数据处理
本章节详细地介绍TF中多线程处理输入数据的解决方案。首先介绍的TFRecord格式可以统一不同的原始数据格式，并更加有效地管理不同的属性。然后介绍如何对图像数据进行预处理，这一节将列举TF支持的图像处理函数，并介绍如何使用这些处理方式来弱化与图像识别无关的因素。复杂的图像处理函数有可能降低训练的速度，为了加速数据预处理过程，后面讲完整地介绍TF多线程数据预处理流程。这一节介绍TF多线程和队列的概念，这是TF多线程数据预处理的基本组成部分。然后讲具体介绍数据预处理流程中的每个部分。最后给出一个完整的多线程数据预处理流程图和程序框架。

## 6.1 TFRecord输入数据格式
TF提供了一种统一的格式来储存数据，这个格式就是TFRecord,TFRecord文件中的数据都是通过 tf.train.Example Protocol Buffer的格式存储的。下面的代码给出了 tf.train.Example的定义。
```
message Example {
    Features features =1;
}

message Features {
    map<string,Feature> feature=1;
}

message Feature{
    oneof kinf{
        BytesList bytes_list =1;
        FloatList folat_list =2;
        Int64List int64_list =3;
    }
}
```
从以上代码可以看出 tf.train.Example的数据结构是比较简洁的。tf.traian.Example中包含了 **一个从属性名称到取值的字典**。其中属性名称为一个字符串，属性的取值可以为字符串(BytesList),实数列表(FloatList)或者整数列表(Int64List).比如讲一张解码前的图像存为一个字符串，图像所对应的类别编号存为整数列表。

## 6.2 TFRecord样例程序
本小节讲给出具体的样例程序来读写TFRecord文件，：
```python
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import matplotlib.pyplot as plt

#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_List=tf.train.Int64List(value=[value]))

#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_List = tf.train.BytesList(value = [value]))

mnist = input_data.read_data_sets("/path/to/MNIST_data/",dtype = tf.uint8,one_hot = True)

images = mnist.train.images

#训练数据所对应的正确答案，可以作为一个属性保存在TFRecord中
labels = mnist.train.labels

#训练数据的图像分辨率，这可以作为Example中的一个属性
pixels = images.shape[1]
num_example = mnist.train.num_example

#输出TFRecord文件的地址
filename = "path/to/output.tf.records"

#创建一个writer来写TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)

for index in range(num_example):
    #将图像矩阵转化为一个字符串
    image_raw = images[index].tostring()

    #将一个样例转化为 Examole Protocol Buffer,并将所有的信息写入这个数据结构
    example = tf.trian.Examole(features = tf.train.Features(feature  = {
        'pixels':_int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)

    }))

    #将一个Example写入TFRecord文件
    writer.write(example.SerializerToString())
writer.close()
```
```python
import tensorflow as tf 

#创建一个reader来读取TFRecord文件中的样例
reader = tf.TFRecordReader()

#创建一个队列来维护输入文件列表。
filename_queue = tf.train.string_input_producer(
    ["/path/to/output.tfrecords"]
)

#从文件中读取一个样例。也可以使用 read_up_to 函数一次性读取多个样例
_,serialized_example = reader.read(filename_queue)
#解析读入的一个样例，如果需要解析多个样例，可以用 parse_example函数
features = tf.parse_single_example(
    serialized_example,
    features = {
        #TF提供两种不同的属性解析方法，一种方法是 tf.FixedLenFeature,这种方法
        #解析的结果是一个 Tensor.另外一种方法是 tf.VarLenFeature.这种方法得到的
        #解析结果为SparseTensor.用于处理稀疏数据。这里解析数据的格式需要和上面程序写入数据
        #的格式一致
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels': tf.FixedLenFeature([],tf.int64),
        'label': tf.FixedLenFeature([],tf.int64)
    }
)

#tf.decode_raw可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

sess = tf.Session()

#启动多线程处理输入数据，7.3节将更加详细地介绍TF多线程处理
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

#每次运行可以去读TFRecord文件中的一个样例，当所有样例都读完之后，在此样例中
#程序会在重头读取
for i in range(10):
    image,label,pixel = sess.run([images,labels,pixels])
```
## 6.3 图像数据处理


### 6.3.1 TF图像处理函数

##### 6.3.1.1 图像编码处理
一张RGB色彩模式的图像可以看成一个三维矩阵，矩阵中的每一个数表示了图像上的不同位置，不同颜色的亮度。然而图像在存储时并不是直接
记录这些矩阵的数字，而是记录经过压缩编码之后的结果。所以要将一张图像还原成一个三维矩阵，需要 **解码**的过程。TF提供了对jpeg和png格式图像的编码/解码函数。：
```python
import tensorflow as tf 
import matplotlib.pyplot as plt 

#读取原生图像的数据
image_raw_data = tf.gfile.FastGFile("./Images/Abyssinian_9.jpg",'rb').read()

with tf.Session() as sess:
    #将图像使用jpeg的格式解码从而得到图像对应的三维矩阵。TF还提供了
    #tf.image.decode_png函数对png格式的图像进行解码，解码之后的结果为
    #张量，在使用它的取值之前需要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data,3,1)

    print(img_data.eval())

    #使用pyplot工具可视化得到的图像。
    plt.imshow(img_data.eval())
    plt.show()

    # 将数据的类型转化为实数方便下面的样例程序对图像进行处理
    img_data = tf.image.convert_image_dtype(image=img_data,dtype=tf.float32)

    #将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中，打开这张图像，
    #可以得到和原始图像一样的图像
    encod_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("./Images/9.jpg","wb") as f:
        f.write(encod_image.eval())
```

##### 6.3.1.2 图像大小调整
一般来说，网络上获取的图像大小是不固定的，但是神经网络输入节点的个数是固定的。所以在将图像的像素作为输入提供给神经网络之前，需要将图像的大小统一。这就是图像大小调整需要完成的任务。图像大小调整有两种方式，第一种是通过算法使得新的图像尽量保存原始图像上的所有信息。TF提供了四种不同的方法，并且将它们封装到了 tf.image.resize_images 函数。
```python
    # 将数据的类型转化为实数方便下面的样例程序对图像进行处理
    # img_data = tf.image.convert_image_dtype(image=img_data,dtype=tf.float32)

    #将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中，打开这张图像，
    #可以得到和原始图像一样的图像
    encod_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("./Images/9.jpg","wb") as f:
        f.write(encod_image.eval())

    #通过tf.image.resize_images函数调整图像的大小，这个函数第一个参数为原始图像
    #第二个和第三个参数为调整后的图像的大小，method参数给出了调整图像大小的方法
    resized = tf.image.resize_images(img_data,[300,300],method=0)

    #输出调整后图像的大小，此处的结果为(300,300,?),图像深度在未明确设定之前是问号
    
    print(resized.get_shape())
    resized = np.asarray(resized.eval(),dtype = 'uint8')
    plt.imshow(resized)
    plt.show()
```
除了将整张图像信息完整保存起来，TF还提供了API对图像进行裁剪或者填充，：
```python
    #通过tf.image.resize_image_with_crop_or_pad 函数调整图像的大小，这个函数的
    #第一个参数为原始图像，后面的两个参数是调整后的目标图像大小。如果原始图像的尺寸
    #大于目标图像，那么这个函数会自动在原始图像的四周填充全0背景。
    croped = tf.image.resize_image_with_crop_or_pad(img_data,300,300)
    padded = tf.image.resize_image_with_crop_or_pad(img_data,700,700)
    print(croped.get_shape())
    print(padded.get_shape())
    croped = np.asarray(croped.eval(),dtype='uint8')
    padded = np.asarray(padded.eval(),dtype='uint8')

    plt.imshow(croped)
    plt.show()
    plt.imshow(padded)
    plt.show()
```
![](picture/2019-05-21-15-51-13.png)
TF还支持通过比例调整图像大小，
```python
    #图像通过比例调整图像大小,
    central_cropped = tf.image.central_crop(img_data,0.5)
    plt.imshow(central_cropped.eval())
    plt.show()

    #图像翻转
    #上下翻转
    flipped = tf.image.flip_up_down(img_data)
    plt.imshow(flipped.eval())
    plt.show()

    #左右
    filpped_LR = tf.image.flip_left_right(img_data)
    plt.imshow(filpped_LR.eval())
    plt.show()

    #对角线
    transposed = tf.image.transpose_image(img_data)
    plt.imshow(transposed.eval())
    plt.show()

    #以一定的概率上下翻转图像
    flipped = tf.image.random_flip_left_right(img_data)
    #以一定概率左右反转图像
    flipped_LR = tf.image.random_flip_left_right(img_data)

    plt.imshow(flipped.eval())
    plt.show()
    plt.imshow(flipped_LR.eval())
    plt.show()

    #调整图像的色彩
    adjusted = tf.image.adjust_brightness(img_data,-0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    adjusted = tf.image.adjust_brightness(img_data,0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    #在[-max_delta,max_delta]的范围内随机调整图像的亮度
    adjusted = tf.image.random_brightness(img_data,max_delta=1)
    plt.imshow(adjusted.eval())
    plt.show()

    #调整图像的对比度
    adjusted = tf.image.adjust_contrast(img_data,-5)
    plt.imshow(adjusted.eval())
    plt.show()
```
##### 6.3.1.3 处理标注框
TF提供了一些工具来处理标注框，下面展示了通过 tf.image.draw_bouding_boxes 函数在图像中加入标注框
```python
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("./Images/Abyssinian_9.jpg",'rb').read()



with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    #tf.image.draw_bouding_boxes函数要求图像矩阵中的数字为实数，所以需先将图像矩阵转化为
    #实数类型。tf.image.draw_bouding_boxes函数图像的输入是一个 batch的数据，也就是多张图像
    #组成的 四维矩阵，所以需要将解码后的图像矩阵加一维。
    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data,tf.float32),0
    )

    #给出每一张图像的所有标注框，一个标注框有四个数字，分别代表[ymin,xmin,ymax,xmax]
    #注意这里给出的数字都是图像的相对位置，比如在 180*267的图像中
    #[0.35,0.47,0.5,0.56]-->[63,125]到[90,150]的图像
    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    result = tf.image.draw_bounding_boxes(batched,boxes)
    result = tf.image.convert_image_dtype(result,tf.uint8)
    result = sess.run(result)
    bat,h,w,c = result.shape
    result = result.reshape(h,w,3)
    plt.imshow(result)
    plt.show()
```
和随机翻转图像、随机调整颜色类似，随机截取图像上有信息含量的部分也是一个提高模型健壮性(robustness)的一种方式。可以使训练得到的模型不受被识别物体大小的影响。通过 tf.image.sample_distorted_bounding_box完成
```python
    begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data),bounding_boxes=boxes
    )
    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data,tf.float32),0
    )

    result = tf.image.draw_bounding_boxes(batched,bbox_for_draw)
    result = tf.image.convert_image_dtype(result,tf.uint8)
    result = sess.run(result)
    bat,h,w,c = result.shape
    result = result.reshape(h,w,3)
    plt.imshow(result)
    plt.show()



    #截取随机出来的图像。
    distorted_image = tf.slice(img_data,begin,size)
    print(distorted_image)
    distorted_image = sess.run(distorted_image)
    print(distorted_image)
    plt.imshow(distorted_image)
    plt.show()
```

### 6.3.2 图像预处理完整案例
```python
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

#给定一张图像，随机调整图像的色彩，因为调整亮度、对比度、饱和度和色相的顺序会影响
#最后得到的记过，所以可以定义多种不同的顺序。具体使用哪一种顺序可以在训练数据预处理时
#随机的选择一种，这样可以进一步降低无关因素对模型的影响

def distort_color(image,color_ordering=0):
    if color_ordering==0:
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
    elif color_ordering ==1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
    elif color_ordering ==2:
        #这里还可以定义其他的排列，
        pass
    return tf.clip_by_value(image,0.0,1.0)#将张量裁剪的指定的最大值和最小值之间

#给定一张解码后的图像、目标图像的尺寸以及图像上的标注框，此函数可以对给出的图像进行预处理。
#这个函数的输入图像是图像识别问题中原始的训练图像，而输出则是神经网络模型的输入层，足以这里只
#处理模型的训练数据，对于预测的数据，一般不需要使用随机变换的步骤

def preprocess_for_train(image,height,width,bbox):
    #如果没有提供标注框，则认为整个图现象就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    
    #转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    #随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),bounding_boxes=bbox
    )
    distorted_image = tf.slice(image,bbox_begin,bbox_size)
    #将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的。
    distorted_image = tf.image.resize_images(distorted_image,[height,width],method=np.random.randint(4))
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image,np.random.randint(2))
    return distorted_image

image_raw_data = tf.gfile.GFile("./Images/Abyssinian_9.jpg",'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    for i in range(6):
        #将图像的尺寸调整为299*299
        result = preprocess_for_train(img_data,299,299,boxes)
        # result = tf.image.convert_image_dtype(result,dtype=tf.uint8)
        print(result)
        result = sess.run(result)
        print(result)
        # plt.imshow(sess.run(result))
        # plt.show()
```
## 6.4 多线程输入数据处理框架
### 6.4.1 python的多线程研究
多任务可以由 **多进程**完成，也可以由一个进程内的 **多线程**完成。进程由若干线程组成的，一个进程至少有一个线程。
由于线程是操作系统直接支持的执行单元，因此，高级语言都内置多线程的支持，python也不例外，并且，python的线程是真正的Posix Thread,而不是模拟出来的线程。
启动一个线程就是把一个函数传入并创建Thread实例，然后调用start()开始执行。
```python
import time,threading

#新线程执行的代码
def loop():
    print("thread %s is runing..."%threading.current_thread().name)
    n=0
    while n<5:
        n=n+1
        print("thread %s >>> %s"%(threading.current_thread().name,n))
        time.sleep(1)
    print("thread %s ended"%threading.current_thread().name)

print("thread %s is running..."%threading.current_thread().name)
t = threading.Thread(target = loop,name='LoopThread')
t.start()
t.join()
print("thread %s ended"%threading.current_thread().name)
'''
thread MainThread is running...
thread LoopThread is runing...
thread LoopThread >>> 1
thread LoopThread >>> 2
thread LoopThread >>> 3
thread LoopThread >>> 4
thread LoopThread >>> 5
thread LoopThread ended
thread MainThread ended
'''
```
由于任何进程默认会启动一个线程，我们把该线程称之为 **主线程**，主线程又可以启动新的线程，python的threading模块有current_thread()函数，它永远返回当前线程的实例，主线程的实例的名字叫做 MainThread,子线程的名字在创建时指定，我们用 LoopThread命名子线程，名字仅仅在打印时用来显示，完全没有其他意义，如果不起名字pyhton就自动给线程命名为 Thread-1,Thread-2.....

> LOCK

多线程和多进程最大的不同在于，多进程中，同一个变量，各自有一份拷贝存在于每个进程中，互不影响，而多线程中，所有变量都由所有线程共享，所以，任何一个变量都可以被任何一个线程修改，因此，线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。
```python
import time,threading

#假定这个是你的银行存款
balance = 0

def change_it(n):
    #先存后取,结果为0
    global balance
    balance = balance +n
    balance = balance -n

def run_thread(n):
    for i in range(10000000):
        change_it(n)

t1 = threading.Thread(target=run_thread,args = (5,))
t2 = threading.Thread(target = run_thread,args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)#???
```
```python
import time
import threading

# 假定这个是你的银行存款
balance = 0
lock = threading.Lock()


def change_it(n):
    # 先存后取,结果为0
    global balance
    balance = balance + n
    balance = balance - n


def run_thread(n):
    for i in range(10000000):
        lock.acquire()
        try:
            change_it(n)
        finally:
            lock.release()
            
t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)

```
当多个线程同时执行lock.acquire()时，只有一个线程能成功地获取锁，然后继续执行代码，其他线程就继续等待直到获取锁为止。获取锁的线程用完后一定释放锁，否者会卡死其他线程。
由于python的GIL锁的存在，导致多核cpu不能有效的利用起来，因此需要采用多 **进程**运行。
```python
            p.apply_async(action,args=(outname,inname,num))

    p.close()
    p.join()
```

### 6.4.2 TF
在上节介绍了使用TF对图像数据进行预处理的方法。虽然使用这些图像数据预处理的方法可以减小无关因素对图像识别模型效果的影响，但这些复杂的预处理过程也会减慢整个训练过程。为了避免图像预处理成为神经网络模型的训练效率的瓶颈，tF提供了一套 **多线程**处理输入数据的框架。
![](picture/2019-05-22-16-42-32.png)
在TF中，队列不仅是一种数据结构，它更提供可多线程机制，队列也是TF多线程输入数据处理框架的基础。

### 6.4.3 队列和多线程
在TF中，队列和变量类似，都是计算图上有状态的节点。其他的计算节点可以修改它们的状态，对于 **变量**，可以通过赋值操作修改变量的取值。对于 **队列**，修改队列状态的操作主要有Enqueuq、EnqueueMant和Dequeue，以下程序展示了如何使用这些函数来操作一个队列。
```python
import tensorflow as tf 

# #创建一个先进先出的队列，指定队列中最多可以保存两个元素，并指定
# #类型为整数
# q = tf.FIFOQueue(capacity=2,dtypes=tf.int32)

# #使用enqueue_many 函数初始化队列中的元素，和变量初始化类似，在使用队列之前
# #需要明确的调用这个初始化过程
# init = q.enqueue_many(([0,10],))

# #使用 Dequeue 函数将队列中的第一个元素出队列，这个函数的值将被存在变量x中。
# x = q.dequeue()
# #将得到的值加1
# y = x+1
# #将加1 的值再重新加入队列中
# q_inc = q.enqueue([y])

# with tf.Session() as sess:
#     #运行处死花花队列的操作
#     init.run()
#     for _ in range(7):
#         # 运行q_inc将执行数据出队列，出队的元素+1，重新加入队列的整个过程。
#         v,_=sess.run([x,q_inc])
#         #打印出队元素的取值
#         print(v) #0  10  1  11  2

##############################################################
'''
input_data = [ [3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
q = tf.FIFOQueue(3,dtypes=[tf.float32])
# init = q.enqueue_many(input_data) #3.0 2.0 1.0
init = q.enqueue(input_data)
out_data = q.dequeue()

with tf.Session() as sess:
    sess.run(init)
    sess.run(init)
    sess.run(init)

    print("1: ",sess.run(out_data))
    print("2: ",sess.run(out_data))
    print("3: ",sess.run(out_data))
    sess.run(q.close(cancel_pending_enqueues=False))
    print(sess.run(q.is_closed()))
# 1:  [3. 2. 1.]
# 2:  [3. 2. 1.]
# 3:  [3. 2. 1.]
# True
'''
###############################################################
'''
input_data = [ [3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
input_data1 = [ [33.,22.,11.],[11.,22.,33.],[111.,222.,333.]]
q = tf.FIFOQueue(3,dtypes=[tf.float32])
init = q.enqueue(input_data)
init1 = q.enqueue(input_data1)

output_data = q.dequeue()
with tf.Session() as sess:
    init.run()
    init1.run()
    init1.run()

    print('1：',sess.run(output_data))
    print('2：',sess.run(output_data))
    print('3：',sess.run(output_data))
    sess.run(q.close(cancel_pending_enqueues=True))
    print(sess.run(q.is_closed()))
# 1： [3. 2. 1.]
# 2： [33. 22. 11.]
# 3： [33. 22. 11.]
# True
'''
#####################################################
'''
# enqueue()每次入列一个元素，对于同一个输入数据，多次入列会重复入列相同的元素
input_data = [ [3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
# q = tf.FIFOQueue(3,dtypes=[tf.float32,tf.float32],shapes=[[],[]])
# 1:  [3.0, 11.0]
# 2:  [2.0, 22.0]
# 3:  [1.0, 33.0]
# True

q = tf.FIFOQueue(3,dtypes = [tf.float32],shapes = [[]])
# 1:  3.0
# 2:  2.0
# 3:  1.0
# True

init = q.enqueue_many(input_data) #3.0 2.0 1.0
# init = q.enqueue(input_data)
out_data = q.dequeue()

with tf.Session() as sess:
    sess.run(init)
    # sess.run(init)
    # sess.run(init)

    print("1: ",sess.run(out_data))
    print("2: ",sess.run(out_data))
    print("3: ",sess.run(out_data))
    sess.run(q.close(cancel_pending_enqueues=False))
    print(sess.run(q.is_closed()))
'''
#################################################################
'''
# enqueue()每次入列一个元素，对于同一个输入数据，多次入列会重复入列相同的元素
input_data = [ [3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
q = tf.FIFOQueue(3,dtypes=[tf.float32,tf.float32],shapes=[[],[]])
init = q.enqueue_many(input_data) 

out_data = q.dequeue_many(2)

with tf.Session() as sess:
    sess.run(init)
    # sess.run(init)
    # sess.run(init)

    print("1: ",sess.run(out_data))
    # print("2: ",sess.run(out_data))
    # print("3: ",sess.run(out_data))
    sess.run(q.close(cancel_pending_enqueues=False))
    print(sess.run(q.is_closed()))

# 如上例队列的元素为 [3.0, 11.0]，[2.0, 22.0]，[1.0, 33.0]，
# dequeue_many(2)将队列元素的第0维3.，2.，1.中取2个数据组成[3., 2.]，然后11.，22.，33.变为第0维，
# 再取2个数据组成[11., 22.]，所以出列的数据为[[3., 2.],[11., 22.],]
'''
#####################################################################

import tensorflow as tf

input_data=[[[3.,2.,1.],[11.,22.,33.],[111.,222.,333.]],[[23.,22.,21.],[211.,222.,233.],[2111.,2222.,2333.]]]
print(tf.shape(input_data))
q=tf.FIFOQueue(3,tf.float32)
init=q.enqueue_many(input_data)
output_data=q.dequeue()
with tf.Session() as sess:
    init.run()
    print('1：',sess.run(output_data))
    print('2：',sess.run(output_data))
    print('3：',sess.run(output_data))
    sess.run(q.close(cancel_pending_enqueues=True))
    print(sess.run(q.is_closed()))

'''
1： [3. 2. 1.]
2： [11. 22. 33.]
3： [111. 222. 333.]
True
输入数据input_data=[[[3.,2.,1.],[11.,22.,33.],[111.,222.,333.]],[[23.,22.,21.],[211.,222.,233.],[2111.,2222.,2333.]]]
为三维，所有第0维的张量为[3. 2. 1.]，[11. 22. 33.]， [111. 222. 333.]，enqueue_many会将他们组合在一起输入队列，
所以队列的元素为[3. 2. 1.]，[11. 22. 33.]， [111. 222. 333.]
'''
```
TF中提供了 **FIFOQueue** 和 **RandomShuffleQueue** 两种队列。后者会将队列中的元素打乱，每次出队列操作得到的是从当前队列元素中随机选择的一个。在训练神经网络时夕阳每次使用的训练数据尽量随机，RandomShuffleQueue提供了这样的功能。
在TF中，队列不仅仅是一种数据结构，还是 **异步计算**张量取值的一个重要的机制，比如多个线程可以同时向同一个队列中写元素，或者同时读取一个队列中的元素。
TF提供了 tf.Coordinator 和 tf.QueueRunnner 两个类来完成多线程协同的功能。tf.Coordinator主要协同多个线程一起停止，提供了 should_stop，request_stop join三个函数，在启动线程前，需要先声明一个 tf.Coordinator类，并将这个类传入每一个创建的线程中，启动的线程需要一直查询 tf.Coordinator类中提供的 should_stop函数，当这个函数的返回值为True时，则当前线程也需要退出。每一个启动的线程都可以通过调用requese_stop函数来通知其他线程退出：
```python
'''
import tensorflow as tf 
import numpy as np 
import threading
import time

#线程中运行的程序，这个程序每隔1秒判断是否需要停止并打印自己的ID
def Loop(coord,worker_id):
    #使用tf.Coordinator 类提供的协同工具判断线程是否需要停止
    while not coord.should_stop():
        #随机停止所有的线程
        if np.random.rand() < 0.1:
            print("stoping id: %d"%worker_id)
            #调用coord.request_stop()函数来通知其他线程停止
            coord.request_stop()
        else:
            #打印当前id
            print("working on id: %d"%worker_id)
        time.sleep(1)

coord = tf.train.Coordinator()

threads = [threading.Thread(target=Loop,args=(coord,i)) for i in range(5)]

for t in threads:t.start()
coord.join(threads)
'''
###########################################################################################
import tensorflow as tf 

#声明一个先进先出的队列，最多100个元素
queue = tf.FIFOQueue(100,"float")
#定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

#使用 tf.train.QueueRunner来创建多个线程运行队列的入队操作
# tf.train.QueueRunner的第一个参数给出了倍操作的列表 [enqueue_op]*5
#表示了需要启动5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)

#将定义过的 QueueRunner加入TF计算图上指定的集合
# tf.train.add_queue_runner函数没有指定集合，
# 则加入默认集合 tf.GraphKeys.QUEUE_RUNNERS.下面的函数就是刚刚定义的qr加入
tf.train.add_queue_runner(qr)

#定义出队操作
out_tensor= queue.dequeue()

with tf.Session() as sess:
    # 使用 tf.train.Coordinator来协同启动的进程
    coord = tf.train.Coordinator()
    #使用 tf.train.QueueRunner时，需要明确调用 tf.train.start_queue_runners
    #来启动所有线程，否者因为没有线程运行入队操作，当调用出队操作时，程序会一直等待入队
    #操作被执行。tf.train.start_queue_runners函数会默认启动 tf.GraphKeys.QUEUE_RUNNERS集合中
    #的所有 QueueRunner. 因为这个函数支支持启动集合中的 QueueRunner，所以一般来说 tf.train.add_queue_runner
    # 函数和 tf.train.start_queue_runners 函数会指向同一个集合
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    #获取队列中的取值
    for _ in range(3):print(sess.run(out_tensor)[0])

    #使用tf.train.Coordinator来停止所有的线程
    coord.request_stop()
    coord.join(threads)
    '''
    上面的程序将启动五个线程来执行队列入队的操作，每个线程都是将随机数写入队列，
0.7624912
-0.8749699
-1.698716
'''
```
### 6.4.2 输入文件队列
https://zhuanlan.zhihu.com/p/27238630
本小节将介绍如何使用TF中的队列管理输入文件列表。TF可以将数据分为多个 TFRecord文件来提高处理频率。TF提供了tf.train.match_filenames_once函数来获取符合一个正则表达式的所有文件，得到的文件列表可以通过 tf.train.string_input_prodecer函数进行有效管理。
tf.train.string_input_prodecer函数会使用初始化时提供的文件列表创建一个队列，输入队列中原始的元素为文件列表的所有文件，创建好的输入队列可以作为文件读取函数的参数。每次调用文件读取函数时，该函数会先判断当前是否已有打开的文件可读，如果没有或者打开的文件已经读完，这个函数会从输入队列中出队一个文件并从这个文件中读取数据。
通过设置 shuffle参数，tf.train.string_input_producer 函数支持随机打乱文件列表中文件出队的顺序，当shuffle参数为True时，文件在加入队列之前会被打乱顺序，所以出队的顺序也是随机的。随机打乱文件顺序以及加入输入队列的过程会跑在一个单独的线程上，这样不会影响文件的速度， tf.train.string_input_prodecer 生成的输入队列可以同时被多个文件读取线程操作，而且输入队列会将队列中的文件均匀地分给不同的线程，不出现有些文件被处理过多次而有些文件还没有被处理过的情况。
当一个输入队列中的文件都被处理完后，它会将初始化提供的文件列表中的文件全部重新加入队列。 tf.train.string_input_prodecer 函数可以设置 num_epoches 参数来限制加载初始文件列表的最大轮数。当所有的文件都被已经被使用了设定的轮数后，如果继续尝试读取新的文件，输入队列会报 OutOfRange 的错误。在测试 神经网络模型时，因为所有的测试数据只需要使用依次，可以将 num_epoches 设置为1.这样在计算完一轮之后程序将自动停止。下面先生成两个 tfrecord:
```python
import tensorflow as tf 

#创建 TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


#模拟海量数据情况下将数据写入不同的文件。num_shards 定义了总共写入多少个文件
# instances_per_shard 定义了每个文件中有多少个数据
num_shards = 2
isinstances_per_shard =2

for i in range(num_shards):
    #将数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分。其中m
    #表示了数据总共被存在了多少个文件中，n表示了当前文件的编号。式样的方式既方便了
    #通过正则表达式获取文件列表，又在文件中加入了更多的信息。
    filename = ('./path/to/data.tfrecords-%.5d-of-%.5d'%(i,num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    #将数据封装成EXample结构并写入TFRECOed格式
    for j in range(isinstances_per_shard):
        #Example结构仅包含当前样例属兔第几个文件以及当前文件的第几个样本
        example = tf.train.Example(features= tf.train.Features
        (feature = {
            'i':_int64_feature(i),
            'j':_int64_feature(j)
        }))
        writer.write(example.SerializeToString())
    writer.close()



#####################################################################################################
# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
# print(dataset1.output_types)  # ==> "tf.float32"
# print(dataset1.output_shapes)  # ==> "(10,)"



# dataset2 = tf.data.Dataset.from_tensor_slices(
#    (tf.random_uniform([4]),
#     tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
# print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
# print(dataset2.output_shapes)  # ==> "((), (100,))"

# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
# print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
# print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"


#############################################################################################################

```
下面是使用 tf.train.match_filenames_once函数和tf.train_string_input_producer函数
```python
import tensorflow as tf 

#使用tf.train.match_filenames_once函数来获取文件列表
files = tf.train.match_filenames_once("./path/to/data.tfrecords-*")

#通过tf.train.string_input_poduce函数创建输入队列，输入队列中的文件列表为
#tf.train.match_filenames_once 函数获取的文件列表。这里将shuffle参数设为False
#来避免随机打乱读文件的顺序。但一般在解决真实问题时，会将 shuffle参数设置为 true
filename_queue = tf.train.string_input_producer(files,shuffle=False,num_epochs=1)

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features =
    {
        'i':tf.FixedLenFeature([],tf.int64),
        'j':tf.FixedLenFeature([],tf.int64)
    }
)

with tf.Session() as sess:
    #虽然在本程序段中没有声明任何变量，但是用 tf.trian.match_filenames_once函数时需要
    #初始化一些变量
    tf.local_variables_initializer().run()
    print(sess.run(files))
    '''
[b'.\\path\\to\\data.tfrecords-00000-of-00002'
 b'.\\path\\to\\data.tfrecords-00001-of-00002']
    '''

    #声明tf.train.Corrdinator类来协同不同线程，并且启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    #多次执行获取数据的操作
    for i in range(6):
        print(sess.run([features['i'],features['j']]))
        '''
[0, 0]
[0, 1]
[1, 0]
[1, 1]
[0, 0]
[0, 1]
        '''
    coord.request_stop()
    coord.join(threads)
```

### 6.4.2 组合训练数据(batching)
TF提供 **tf.train.batch** ,和 **tf.train.shffle_batch** 函数来将单个的样例组织成batch的形式输出。这两个函数都会生成一个 **队列**。队列的入队操作是生成单个样例的方法，而每次出队得到的是一个 batch的样例。它们唯一的区别在于是否将数据顺序打乱。
```python
   example,label = features['i'],features['j']

    #一个batch中样例的个数
    batch_size =3
    #组合样例的队列中最多可以存储的样例个数，这个队列如果太大，那么需要占用许多内存资源：
    #如果太小，那么出队操作可能会因为没有数据而被阻碍(block),从而导致训练效率降低。一般来说
    #这个队列的大小会和每一个batch的大小有关，下面一行代码给出了设置队列大小的一种方式
    capacity = 1000 + 3*batch_size

    #使用tf.train.batch函数来组合样例，[example,label]参数给出了需要组合的元素，一般example和
    #label分别代表 训练样本和这个样本对应的正确标签。batch_size参数给出了每个batch中样例的个数。
    # capacity给出了队列最大的容量，当队列长度等于容量时，TF将暂停入队操作，而只是等待元素出队。
    #元素个数小于容量时，TF重新启动入队
    example_batch,label_batch = tf.train.batch(
        [example,label],batch_size=batch_size,capacity=capacity
    )

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        #获取打印组合之后的样例。在真实问题中，这个输出一般会作为神经网络的输入
        for i in range(2):
            cur_example_batch,cur_label_batch = sess.run(
                [example_batch,label_batch]
            )
            print(cur_example_batch,cur_label_batch)
        coord.request_stop()
        coord.join(threads)       
```

### 6.4.5 输入数据处理框架
```python
import tensorflow as tf 

#创建文件列表，并通过文件列表创建输入文件队列。在调用输入数据处理流程前，需要
#统一所有原始数据的格式并将它们存储到TFRecoed文件中
files = tf.train.match_filenames_once("./path/to/data.tfrecords-*")
filename_queue =tf.train.string_input_producer(files,shuffle=False)

#计息TFRecoed文件中的数据，这里假设image存储的是图像的原始数据，label为标签
#height，width和channels
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features = 
    {
        'image': tf.FixedLenFeature([],tf.string),
        'label': tf.FixedLenFeature([],tf.int64),
        'height': tf.FixedLenFeature([].tf.int64),
        'width': tf.FixedLenFeature([],tf.int64),
        'channels': tf.FixedLenFeature([],tf.int64),
    }
)
image,label = features['image'],features['label']
height,width = features['height'],features['width']
channels = features['channels']

#从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image,tf.uint8)
decoded_image.set_shape([height,width,channels])
#定义神经网络输入层图片的大小
image_size=299
#图像预处理
distorted_image = prepocess_for_train(
    decoded_image,image_size,image_size,None
)

#将处理后的图像和标签数据通过tf.train.shuffle_batch整理成神经网络训练时
#需要的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3*batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue
)
#定义神经网络结构以及优化过程，image_batch可以作为输入提供给神经网络的输入层
#label_batch则提供了输入batch中样例的正确答案
logit = inference(image_batch)
loss = calc_loss(logit,label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#声明绘画并运行神经网络的优化过程
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    #神经网络训练过程
    for i in range(TRAINING_ROUDNS):
        sess.run(train_step)

    #停止所有线程
    coord.request_stop()
    coord.join()
```

![](picture/2019-05-24-09-58-39.png)

## 6.5 数据集(Dataset)
TF提供一套更高层的数据处理框架，在新的框架中，每一个数据来源被抽象成一个“数据集”，开发者可以以数据集为基本对象，方便地进行batching、随机打乱(shuffle)等操作。从1.3版本起，TF推荐正式使用数据集作为输入数据的首选框架。从1.4版本起，数据集框架从tf.contrib.data迁移到tf.data.成为tf的核心部件。

### 6.5.1 数据集的基本使用方法
在数据集框架中，每一个数据集代表一个数据来源：数据可能来自一个张量，一个TFRcord文件，一个文本文件，或者经过sharding的一系列文件，等等。由于训练数据通常无法全部写入内存中，从数据集中读取数据时需要使用一个迭代器(iterator)按顺序进行读取，这点与队列的dequeue()和Reader的read()操作类似，与队列相似，数据集也是计算图上的一个节点。
如下，从张量创建一个数据集，遍历这个数据集，并对每个输入输出$y=x^2$：
```python
import tensorflow as tf 

#从一个数组创建数据集
input_data = [1,2,3,4,5,6,7,8]
dataset = tf.data.Dataset.from_tensor_slices(input_data)

#定义一个迭代器用于遍历数据集。因为上面定义的数据集没有用 palceholder
#作为输入参数，所以这里可以使用最简单的 one_shot_iterator
iterator = dataset.make_one_shot_iterator()

# get_next()返回表示一个输入数据的张量，类似于队列的 dequeue()
x = iterator.get_next()
y = x*x

with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))
'''
1
4
9
16
25
36
49
64
'''
```
从以上的例子可以看到，利用数据集读取数据有三个基本步骤。
1. 定义数据集的构造方法
这个例子使用了 tf.data.Dataset.from_tensor_slices()，表明数据集是从一个张量中构建的，如果数据集是从文件中构建的，则需要相应调用不同的构造方法。**TFRecoedDataset** **TextLineDataset**
2. 定义遍历器
这里例子使用了最简单的 one_shot_iterator 来遍历数据集。稍后将介绍更加灵活的 initializable_iterator
3. 使用 get_next() 方法从遍历器中读取数据张量，作为计算图的其他部分的输入


在真实的项目中，训练数据通常时保存在硬盘文件中，这时可以用TextLineDataset来更加方便的读取数据。
```python
import tensorflow as tf 

#从文本文件创建数据集。假定每行文字时一个训练例子，注意这里可以提供多个文件。
# files = tf.train.match_filenames_once('G:\许的数据集\csv1\\gc*')
# dataset = tf.data.TextLineDataset(files)

dataset = tf.data.TextLineDataset(['G:\许的数据集\csv1\\gc1.csv','G:\许的数据集\csv1\\gc2.csv'])

#定义迭代器用于遍历数据集
iterator = dataset.make_one_shot_iterator()

#这里get_next()返回一个字符串类型的张量，代表文件中的一行

x = iterator.get_next()
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    # print(sess.run(files))
    for i in range(3):
        print(sess.run(x))
```
在图像相关任务中，输入数据通常以TFRecord形式存储，这时可以用 **TFRecoedDataset** 来读取数据，与文本文件不同，每一个TFRecord都有自己不同的 feature格式，因此在读取 TFRecord时，需要提供一个 **parser** 函数来解析所读取的 TFRecord的数据格式。
```python
import tensorflow as tf 
#解析一个 TFredord的方法。record是从文件中读取的一个样例。7.1节中具体介绍了如何解析TFrecord样例
#如何解析 TFRecord样例
def parser(record):
    #解析读入的一个样例
    features = tf.parse_single_example(
        record,
        features = {
            'i': tf.FixedLenFeature([],tf.int64),
            'j': tf.FixedLenFeature([],tf.int64)
        }
    )
    return features['i'],features['j']

#从 TFRecord 文件创建数据集
# files = tf.train.match_filenames_once("G:\Code\study_tensorflow_google\\tree_book\chapter6\path\\to\\")
dataset = tf.data.TFRecordDataset("G:\Code\study_tensorflow_google\\tree_book\chapter6\path\\to\data.tfrecords-00000-of-00002")

#map()函数表示对数据集中的每一条数据进行调用相应方法。使用 TFRecordDataset读出的
#是二进制的数据，这里需要通过map() 来调用 parse() 对二进制数据进行解析。类似地，
#map()函数也可以用来完成其他的数据预处理工作
dataset = dataset.map(parser)

#定义遍历数据集的迭代器
iterator = dataset.make_one_shot_iterator()

#feat1，feat2是 parser()返回的 一维的 int64型张量，可以作为输入用于进一步的计算
feat1,feat2 = iterator.get_next()

with tf.Session() as sess:
    # tf.local_variables_initializer().run()
    for i in range(2):
        f1,f2 = sess.run([feat1,feat2])
        print(f1,f2)
```
以上例子使用了最简单的 **one_shot_iterator** 来遍历数据集，在使用 **one_show_iterator**时，数据集的所有参数必须已经确定，因此 **one_show_iterator** **不需要特别的初始化过程**。如果需要用 placeholder 来初始化数据集，那就需要用到 initializeble_iterator.以下代码给出了用 **initializeble_iterator** 来动态初始化数据集的例子
```python
import tensorflow as tf 

#解析一个TFRecord的方法
def parser(record):
    features = tf.parse_single_example(
        record,
        features = {
            'i':tf.FixedLenFeature([],tf.int64),
            'j':tf.FixedLenFeature([],tf.int64),
        }
    )
    return features['i'],features['j']

files = tf.train.match_filenames_once("G:\Code\study_tensorflow_google\\tree_book\chapter6\path\\to\*")
input_files = tf.placeholder(tf.string) 
dataset = tf.data.TFRecordDataset(input_files)
dataset= dataset.map(parser)

#定义遍历dataset的initializeble_iterator
iterator = dataset.make_initializable_iterator()
feat1,feat2 = iterator.get_next()

with tf.Session() as sess:
    #首先初始化interator,并给出input_files的值
    tf.local_variables_initializer().run()
    print(files)
    sess.run(iterator.initializer,
    feed_dict = {
        input_files:files.eval
    })
    #遍历所有数据一个 epoch.当遍历结束时，程序会抛出OutofrangeError
    while True:
        try:
            f1,f2 = sess.run([feat1,feat2])
            print(f1,f2)
        except tf.errors.OutOfRangeError:
            break
```
### 6.5.1 数据集的高层操作
>dataset = dataset.map(parser)

**map** 是在数据集上进行操作的最常用的方法之一，在这里，map(parser)方法对数据集中的每一条数据调用参数指定的parser方法。对每一条数据进行处理后，map将处理后的数据包装成一个新的数据集返回，map函数非常灵活，可以用于对数据的预处理操作
>distorted_image = preprocess_for_train(
    decode_image,image_size,image_size,None
)

而在数据集框架中，可以通过map来对每一条数据调用preprocess_for_train方法：
>dataset = dataset.map(lambda x : preprocess_for_train(x,image_size,image_size,None))

在上面的代码中，lambda表达式的作用是将原来有的4个参数的函数转化为只有一个参数的函数。preprocess_for_train函数的第一个参数decoded_image变成了lambda表达式中的x，这个参数就是原来函数中的参数 decoded_image. preprocess_for_train函数中后三个参数都被替换成了具体的数值。注意这里的image_size是一个变量，有具体取值，该值需要在程序的上文给出。
从表面上看，新的代码在长度上似乎并没有缩短，然而由于map方法返回的是一个 **新的数据集**，可以直接继续调用其他高层操作。在上一节介绍的队列框架中，预处理，shuffle，batch等操作有的在都队列上进行，有的在图片张量上进行，整个代码处理流程在处理队列和张量的代码片段中来回切换。而在数据集操作中，所有的操作都在数据集上进行，这样的代码结构将非常干净，简洁。
**tf.train.batch**和 **tf.train.shuffle_batch**方法，在数据集框架中，shuffle 和 batch 操作由两个方法独立实现：
>dataset = dataset.shuffle(buffer_size) #随机打乱顺序
>dataset = dataset.batch(batch_size)#将数据组合成batch

其中shuffle的方法的参数 buffle_size等效于 tf.train.shuffle_batch的 min_after_dequeue参数。shuffle算法在内部使用一个缓冲区保存 **buffer_size** 条数据，每读入一条新数据时，从这个缓冲区中选择出一条数据进行输出。缓冲区的大小越大，随机的性能越好，但占用的内存也就越多。
**batch** 方法的参数 batch_size 代表要输出的每个 batch由多少条数据组成。如果数据集中包含多个张量，那么batch操作将对每一个张量分开进行，举例而言，如果数据集中的每一个数据(即 iterator.get_next()返回的值)是 image、label两个张量，其中image的维度是[300,300],label的维度是 [],batch_size是128，那么经过batch操作后的数据集的每一个输出将包含两个维度分别是 [128,300,300] 和[128]的张量。
**repeat**是另一个常用的操作方法。这个方法将数据集中的数据复制多份，其中每一份的数据被称为一个 epoch.
>dataet = dataet.repeat(N) #将数据集重复N份

需要指出的是，如果数据集在 **repeat** 前已经进行了 shuffle操作，输出的每一个 epoch中随机 shuffle的结果并不会相同。例如，如果输入的数据是 [1,2,3],shuffle后输出的第一个epoch是 [2,1,3],而第二个 epoch则可能是 [3,2,1].**repeat**和 **map**、**shuffle**、**batch**等操作一样，都只是计算图中的一个计算节点，repeat只代表重复相同的处理过程，并不会记录前一 epoch的处理结果。
除了这些方法，数据集还提供了其他多种操作。例如，**concatenate()**将两个数据集顺序连接起来，**take(N)** 从数据集读取前N项数据。 **skip(N)**在数据集中跳过前N项数据， **flap_map()**从多个数据集中轮流读取数据，等等。

```python
import tensorflow as tf 

#列举输入文件，训练和测试使用不同的数据
train_files = tf.train.match_filenames_once("/path/to/train_file-*")
test_files = tf.train.match_filenames_once("/path/to/test_file-*")

#定义parser方法从TFRecord中解析数据。这里假设 image中存储的是图像的原始数据
# label 为该样例所对应的标签，height,width和channels给出了图片的维度
def parser(record):
    features = tf.parse_single_example(
        record,
        featires = {
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64),
            'height':tf.FixedLenFeature([],tf.int64),
            'width':tf.FixedLenFeature([],tf.int64),
            'channels':tf.FixedLenFeature([],tf.int64),
        }
    )

    #从原始图像数据解析出象素矩阵，并根据图像尺寸还原图像
    decode_image = tf.decode_raw(features['image'],tf.uint8)
    decode_image.set_shape([features['height'],features['width'],features['channels']])
    label = features['label']
    return decode_image,label

image_size = 299 #定义神经网络输入层图片的大小
batch_size = 100 #定义组合数据batch的大小
shuffle_buffer = 10000 #定义随机打乱数据时 buffer的大小

#定义读取训练数据的数据集
dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)

#对数据依次进行预处理，shuffle，和 batching操作。preprocess_for_train 为
#7.2.2小节中介绍的图像预处理程序。因为上一个mao得到的数据集中提供了 decoded_image
#和 label两个结果，所以这个map需要提供一个有2个参数的函数来处理数据，在下面的代码中，
#lambda中的image代表的就是第一个 map返回的 decoded_image,label代表的就是第一个map返回的label.
#在这个lambda表达式中我们首先将decoded_image在传入 proprocess_for_train来进一步对图像数据
#进行预处理。然后再将处理好的图像和label组成最终的输出

dataset = dataset.map(lambda image,label:(
    preprocess_for_train(image,image_szie,image_size,None),label
))
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

#重复Num_EPOCHS个 epoch,再上节 TRAINING_ROUNDS指定了训练的轮数，而在这里指定了
#整个数据集重复的次数，它也间接的确定了训练的轮数
NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)

#定义数据集迭代器，虽然定义数据集时没有直接使用placeholder来提供文件地址，但是
#tf.trian.match_filenames_once 方法得到的结果和placeholder的机制类似
#也需要初始化，所以这里使用的是 initializable_iterator
iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()

#定义神经网络结构以及优化过程
learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit,label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#定义测试用的 Dataset,与训练不同，测试数据的Dataset不需要经过随机翻转等预处理操作，也不需要打乱顺序和
#重复多个 epoch，这里使用与训练数据先用的 parser进行解析，调整分辨率到网络输入层大小，然后直接进行 batching操作
test_dataset - tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(lambda image,label:
tf.image.resize_images(image,[image_size,image_size]),label))
test_dataset = test_dataset.batch(batch_size)

#定义测试数据上的迭代器
test_iterator = test_dataset.make_initializable_iterator()
test_image_batch,test_label_batch = test_iterator.get_next()

#定义预测结果为logit值最大的分类
test_logit = inference(test_image_batch)
predictions = tf.argmax(test_logit,axis=-1,output_type = tf.int32)

#声明会话并运行神经网络的优化过程
with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer(),
    tf.local_variables_initializer())

    #初始化训练数据的迭代器
    sess.run(iterator.initializer)

    #循环进行训练，直到数据集完成输入、抛出 outOfRangeError错误
    while True:
        try:
            sess.run(train_step)
        except: tf.errors.OutOfRangeError:
            break
    
    #初始化测试数据的迭代器
    sess.run(test_iterator.initializer)

    #获取预测结果
    test_results = []
    test_labels = []
    while True:
        try:
            pred,label = sess.run([predictions,test_label_batch])
            test_results.extend(pred)
            test_labels.extend(label)
        except tf.errors.OutOfRangeError:
            break

#计算准确率
correct = [float (y==y_) for (y,y_) in zip (test_results,test_labels)
accuracy = sum(correct) / len(correct)
```