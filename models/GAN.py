import datetime
import tensorflow as tf
from tools.losses import GAN_generator_loss as generator_loss, discriminator_loss
from basic_models.generator import Generator
from basic_models.discriminator import  simple_Discriminator as Discriminator
import os

class GAN():
  def __init__(self):
    self.generator = Generator()
    self.discriminator = Discriminator()

    self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    log_dir="logs/"
    self.summary_writer = tf.summary.create_file_writer(
      log_dir + "GAN_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    self.checkpoint_dir = './training_checkpoints/GAN'
    self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "GAN_ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generator=self.generator,
                                    discriminator=self.discriminator)

  def __str__(self):
    return "GAN"

  def load_ckpt(self):
    self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

  @tf.function
  def train_step(self, input_image, target, epoch):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          gen_output = self.generator(input_image, training=True)

          disc_real_output = self.discriminator(target, training=True)
          disc_generated_output = self.discriminator(gen_output, training=True)

          gen_total_loss = generator_loss(disc_generated_output, gen_output)
          disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              self.generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                  self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                              self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  self.discriminator.trainable_variables))

      with self.summary_writer.as_default():
          tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
          tf.summary.scalar('disc_loss', disc_loss, step=epoch)


