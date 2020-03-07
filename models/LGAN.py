import datetime
import tensorflow as tf
from tools.losses import LGAN_generator_loss as generator_loss, discriminator_loss
from basic_models.generator import Generator
from basic_models.discriminator import  Discriminator as Discriminator
import os

class LGAN():
  def __init__(self):
    self.generator = Generator()
    self.discriminator = Discriminator()

    self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    log_dir="logs/"
    self.summary_writer = tf.summary.create_file_writer(
      log_dir + "LGAN_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    self.checkpoint_dir = './training_checkpoints'
    self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "LGAN_ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generator=self.generator,
                                    discriminator=self.discriminator)

  def __str__(self):
    return "LGAN"

  @tf.function
  def train_step(self, input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = self.generator(input_image, training=True)

      disc_real_output = self.discriminator([input_image, target], training=True)
      disc_generated_output = self.discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
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
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
      tf.summary.scalar('disc_loss', disc_loss, step=epoch)



