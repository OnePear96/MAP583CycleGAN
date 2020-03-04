import datetime
import tensorflow as tf
from tools.losses import L_generator_loss as generator_loss
from basic_models.generator import Generator
import os

class UNN():
  
  def __init__(self):
    super(UNN,self).__init__()
    self.generator = Generator()
    self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    log_dir="logs/"
    self.summary_writer = tf.summary.create_file_writer(
      log_dir + "UNN_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    checkpoint_dir = './UNN_training_checkpoints'
    self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    generator=self.generator)

  def get_generator(self):
    return self.generator

  @tf.function
  def train_step(self, input_image, target, epoch):
    generator, generator_optimizer = self.generator, self.generator_optimizer
    with tf.GradientTape() as gen_tape:
      gen_output = generator(input_image, training=True)
      gen_loss = generator_loss(gen_output, target)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))


    with self.summary_writer.as_default():
      tf.summary.scalar('gen_loss', gen_loss, step=epoch)

