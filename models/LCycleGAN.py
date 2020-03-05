import datetime
import tensorflow as tf
from tools.losses import LCycle_generator_loss as generator_loss, discriminator_loss
from basic_models.generator import Generator
from basic_models.discriminator import  Discriminator as Discriminator
import os

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


class LCycleGAN():

    def __init__(self):
        self.generator_X2Y = Generator()
        self.generator_Y2X = Generator()
        self.discriminator_X = Discriminator()
        self.discriminator_Y = Discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        log_dir="logs/"
        self.summary_writer = tf.summary.create_file_writer(
            log_dir + "LCyclGANfit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        checkpoint_dir = './LCycleGAN_training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer = self.discriminator_optimizer,
                                        generator_X2Y=self.generator_X2Y,
                                        generator_Y2X=self.generator_Y2X,
                                        discriminator_X=self.discriminator_X,
                                        discriminator_Y=self.discriminator_Y)

    @tf.function
    def train_step(self, X, Y, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            D_real_Y_output = self.discriminator_Y([X,Y],training = True)
            D_real_X_output = self.discriminator_X([Y,X],training = True)
            
            fake_Y = self.generator_X2Y(X,training = True)
            D_fake_Y_output = self.discriminator_Y([X,fake_Y],training = True)
            fake_X = self.generator_Y2X(Y,training = True)
            D_fake_X_output = self.discriminator_X([Y,fake_X],training = True)
            
            D_loss = discriminator_loss(D_real_X_output, D_fake_X_output) + discriminator_loss(D_real_Y_output, D_fake_Y_output)
            
            disc_params = list(self.discriminator_X.trainable_variables)+list(self.discriminator_Y.trainable_variables) 
            discriminator_gradients = disc_tape.gradient(D_loss,disc_params)
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc_params))
            
            fake_Y = self.generator_X2Y(X,training = True)
            D_fake_Y_output = self.discriminator_Y([X,fake_Y],training = True)
            reconstructed_X = self.generator_Y2X(fake_Y,training = True)
            generator_X_loss, gan_X_loss, l1_X_loss, reconstructed_X_loss= generator_loss(D_fake_Y_output,fake_Y,Y,X,reconstructed_X)
            
            gen_params = list(self.generator_X2Y.trainable_variables)+list(self.generator_Y2X.trainable_variables)
        
            fake_X = self.generator_Y2X(Y,training = True)
            D_fake_X_output = self.discriminator_X([Y,fake_X],training = True)
            reconstructed_Y = self.generator_X2Y(fake_X,training = True)
            generator_Y_loss, gan_Y_loss, l1_Y_loss, reconstructed_Y_loss = generator_loss(D_fake_X_output,fake_X,X,Y,reconstructed_Y)
            
            gen_total_loss = generator_X_loss+generator_Y_loss
            
            gen_gradients = gen_tape.gradient(gen_total_loss,gen_params)
            self.generator_optimizer.apply_gradients(zip(gen_gradients,gen_params))
            
            with self.summary_writer.as_default():
                tf.summary.scalar('gen_total_loss',gen_total_loss, step = epoch)
                tf.summary.scalar('disc_total_loss',D_loss, step=epoch)




'''


log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "LCyclGANfit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(X, Y, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
  #  gen_X2Y_output = generator_X2Y(X, training=True)
    D_real_Y_output = discriminator_Y([X,Y],training = True)
    D_real_X_output = discriminator_X([Y,X],training = True)
#    D_real_loss = discriminator_real_loss(D_real_X_output)+discriminator_real_loss(D_real_Y_output)
    
    fake_Y = generator_X2Y(X,training = True)
    D_fake_Y_output = discriminator_Y([X,fake_Y],training = True)
    fake_X = generator_Y2X(Y,training = True)
    D_fake_X_output = discriminator_X([Y,fake_X],training = True)
#    D_fake_loss = discriminator_fake_loss(D_fake_Y_output) + discriminator_fake_loss(D_fake_X_output)
    
    D_loss = discriminator_loss(D_real_X_output, D_fake_X_output) + discriminator_loss(D_real_Y_output, D_fake_Y_output)
    
    disc_params = list(discriminator_X.trainable_variables)+list(discriminator_Y.trainable_variables) 
    discriminator_gradients = disc_tape.gradient(D_loss,disc_params)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc_params))
    
    fake_Y = generator_X2Y(X,training = True)
    D_fake_Y_output = discriminator_Y([X,fake_Y],training = True)
    reconstructed_X = generator_Y2X(fake_Y,training = True)
    generator_X_loss, gan_X_loss, l1_X_loss, reconstructed_X_loss= generator_loss(D_fake_Y_output,fake_Y,Y,X,reconstructed_X)
    
    gen_params = list(generator_X2Y.trainable_variables)+list(generator_Y2X.trainable_variables)
   
    fake_X = generator_Y2X(Y,training = True)
    D_fake_X_output = discriminator_X([Y,fake_X],training = True)
    reconstructed_Y = generator_X2Y(fake_X,training = True)
    generator_Y_loss, gan_Y_loss, l1_Y_loss, reconstructed_Y_loss = generator_loss(D_fake_X_output,fake_X,X,Y,reconstructed_Y)
    
    gen_total_loss = generator_X_loss+generator_Y_loss
    
    gen_gradients = gen_tape.gradient(gen_total_loss,gen_params)
    generator_optimizer.apply_gradients(zip(gen_gradients,gen_params))
    
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss',gen_total_loss, step = epoch)
        tf.summary.scalar('disc_total_loss',D_loss, step=epoch)


checkpoint_dir = './LCycleGAN_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator_X2Y=generator_X2Y,
                                 generator_Y2X=generator_Y2X,
                                 discriminator_X=discriminator_X,
                                 discriminator_Y=discriminator_Y)

'''