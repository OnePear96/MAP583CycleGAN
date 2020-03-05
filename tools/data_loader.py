import tensorflow as tf 

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
PATH = 'data'

def resize(image, height, width):
    image = tf.image.resize(image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]

def normalize(image):
    image = (image / 127.5) - 1
    return image


class load_image_s():
    def __init__(self,path = './data/MapAerialSup/', buffer = 400, batch = 1):
        super().__init__()
        self.BUFFER_SIZE = buffer
        self.BATCH_SIZE = batch
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.PATH = path
    
    def load(self,image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        w = tf.shape(image)[1]

        w = w // 2
        input_image = image[:, :w, :]
        real_image = image[:, w:, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image,real_image):
        # resizing to 286 x 286 x 3
        input_image =  resize(input_image, 286, 286)
        real_image = resize(real_image, 286, 286)
        # randomly cropping to 256 x 256 x 3
        input_image, real_image = random_crop(input_image, real_image)
        if tf.random.uniform(()).numpy() > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
        return input_image, real_image


    def load_image_train(self,image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image = normalize(input_image)
        real_image = normalize(real_image)
        return input_image, real_image


    def load_image_test(self,image_file):
        input_image, real_image = self.load(image_file)
        input_image = resize(input_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        real_image = resize(real_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image = normalize(input_image)
        real_image = normalize(real_image)
        return input_image, real_image

    def get_train_set(self):
        train_dataset = tf.data.Dataset.list_files(self.PATH+'train/*.jpg')
        train_dataset = train_dataset.map(self.load_image_train,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.batch(BATCH_SIZE)
        self.train_dataset = train_dataset
        return train_dataset

    def get_test_set(self):
        test_dataset = tf.data.Dataset.list_files(self.PATH+'val/*.jpg')
        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        self.test_dataset = test_dataset
        return test_dataset


class load_image_u():
    def __init__(self,path = './data/MapAerialSup/', buffer = 400, batch = 1, name_X = 'A', name_Y = 'B'):
        super().__init__()
        self.BUFFER_SIZE = buffer
        self.BATCH_SIZE = batch
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.PATH = path
        self.name_X = name_X
        self.name_Y = name_Y
    
    def load(self,image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image, tf.float32)
        return image

    @tf.function()
    def random_jitter(self,input_image):
        # resizing to 64 x 64 x 3
        input_image = resize(input_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        if tf.random.uniform(()).numpy() > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
        return input_image
    
    def load_image_train(self,image_file):
        input_image = self.load(image_file)
        input_image = self.random_jitter(input_image)
        input_image = normalize(input_image)
        return input_image
    
    def load_image_test(self,image_file):
        input_image = self.load(image_file)
        input_image = resize(input_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image = normalize(input_image)
        return input_image

    def get_train_set(self):
        path_train_X = PATH + 'train'+self.name_X
        train_dataset_X = tf.data.Dataset.list_files(path_train_X+'/*.jpg')
        train_dataset_X = train_dataset_X.map(self.load_image_train,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset_X = train_dataset_X.shuffle(self.BUFFER_SIZE)
        train_dataset_X = train_dataset_X.batch(self.BATCH_SIZE)
        self.train_dataset_X = train_dataset_X

        path_train_Y = PATH + 'train'+self.name_Y
        train_dataset_Y = tf.data.Dataset.list_files(path_train_Y+'/*.jpg')
        train_dataset_Y = train_dataset_Y.map(self.load_image_train,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset_Y = train_dataset_Y.shuffle(self.BUFFER_SIZE)
        train_dataset_Y = train_dataset_Y.batch(self.BATCH_SIZE)
        self.train_dataset_Y = train_dataset_Y

        return train_dataset_X, train_dataset_Y

    def get_test_set(self):
        path_test_X = PATH + 'test' + self.name_X
        test_dataset_X = tf.data.Dataset.list_files(path_test_X+'/*.jpg')
        test_dataset_X = test_dataset_X.map(self.load_image_test)
        test_dataset_X = test_dataset_X.batch(self.BATCH_SIZE)
        self.test_dataset_X = test_dataset_X

        path_test_Y = PATH + 'test' + self.name_Y
        test_dataset_Y = tf.data.Dataset.list_files(path_test_Y+'/*.jpg')
        test_dataset_Y = test_dataset_Y.map(self.load_image_test)
        test_dataset_Y = test_dataset_Y.batch(self.BATCH_SIZE)
        self.test_dataset_Y = test_dataset_Y

        return test_dataset_X,test_dataset_Y