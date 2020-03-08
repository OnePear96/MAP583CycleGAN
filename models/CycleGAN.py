import datetime
import tensorflow as tf
from tools.losses import Cycle_generator_loss as generator_loss, discriminator_loss
from basic_models.generator import Generator
from basic_models.discriminator import simple_Discriminator as Discriminator
import os

class CycleGAN():
  def __init__(self, cX = 3, cY = 3):
    self.generator_X2Y = Generator(ic = cX, oc = cY)
    self.generator_Y2X = Generator(ic = cY, oc = cX)
    self.discriminator_X = Discriminator(c = cX)
    self.discriminator_Y = Discriminator(c = cY)

    self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    log_dir="logs/"
    self.summary_writer = tf.summary.create_file_writer(
      log_dir + "CycleGAN_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    self.checkpoint_dir = './training_checkpoints'
    self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "CycleGAN_ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer = self.discriminator_optimizer,
                                    generator_X2Y=self.generator_X2Y,
                                    generator_Y2X=self.generator_Y2X,
                                    discriminator_X=self.discriminator_X,
                                    discriminator_Y=self.discriminator_Y)

  def __str__(self):
    return 'CycleGAN'

  @tf.function
  def train_step(self, X, Y, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #  gen_X2Y_output = generator_X2Y(X, training=True)
      D_real_Y_output = self.discriminator_Y(Y,training = True)
      D_real_X_output = self.discriminator_X(X,training = True)
      
      fake_Y = self.generator_X2Y(X,training = True)
      D_fake_Y_output = self.discriminator_Y(fake_Y,training = True)
      fake_X = self.generator_Y2X(Y,training = True)
      D_fake_X_output = self.discriminator_X(fake_X,training = True)
    #  D_fake_loss = discriminator_fake_loss(D_fake_Y_output) + discriminator_fake_loss(D_fake_X_output)
      
      D_loss = discriminator_loss(D_real_X_output, D_fake_X_output) + discriminator_loss(D_real_Y_output, D_fake_Y_output)
      
      disc_params = list(self.discriminator_X.trainable_variables)+list(self.discriminator_Y.trainable_variables) 
      discriminator_gradients = disc_tape.gradient(D_loss,disc_params)
      self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc_params))
      
      ## Generator

      fake_Y = self.generator_X2Y(X,training = True)
      D_fake_Y_output = self.discriminator_Y(fake_Y,training = True)
      reconstructed_X = self.generator_Y2X(fake_Y,training = True)
      generator_X_loss, gan_X_loss, l1_X_loss = generator_loss(D_fake_Y_output,fake_Y,X,reconstructed_X)
      
      gen_params = list(self.generator_X2Y.trainable_variables)+list(self.generator_Y2X.trainable_variables)
  
      fake_X = self.generator_Y2X(Y,training = True)
      D_fake_X_output = self.discriminator_X(fake_X,training = True)
      reconstructed_Y = self.generator_X2Y(fake_X,training = True)
      generator_Y_loss, gan_Y_loss, l1_Y_loss = generator_loss(D_fake_X_output,fake_X,Y,reconstructed_Y)
      
      gen_total_loss = generator_X_loss+generator_Y_loss
      
      gen_gradients = gen_tape.gradient(gen_total_loss,gen_params)
      self.generator_optimizer.apply_gradients(zip(gen_gradients,gen_params))
      
      with self.summary_writer.as_default():
          tf.summary.scalar('gen_total_loss',gen_total_loss, step = epoch)
          tf.summary.scalar('disc_total_loss',D_loss, step=epoch)
          tf.summary.scalar('gen_total_loss',l1_X_loss+l1_Y_loss, step = epoch)


