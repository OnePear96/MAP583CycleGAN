import datetime
import tensorflow as tf
from tools.losses import GAN_generator_loss as generator_loss, discriminator_loss
from basic_models.generator import Generator
from basic_models.discriminator import  simple_Discriminator as Discriminator


class GAN():
  def __init__(self):
    self.generator = Generator()
    self.discriminator = Discriminator()

    self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    log_dir="logs/"
    self.summary_writer = tf.summary.create_file_writer(
      log_dir + "GAN_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    checkpoint_dir = './GAN_training_checkpoints'
    self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

  @tf.function
  def train_step(self, input_image, target, epoch):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          gen_output = generator(input_image, training=True)

          disc_real_output = discriminator(target, training=True)
          disc_generated_output = discriminator(gen_output, training=True)

          gen_total_loss = generator_loss(disc_generated_output, gen_output)
          disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                  discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))

      with summary_writer.as_default():
          tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
          tf.summary.scalar('disc_loss', disc_loss, step=epoch)


