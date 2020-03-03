import datetime
import tensorflow as tf
from tools.losses import Cycle_generator_loss as generator_loss, discriminator_loss
from basic_models.generator import Generator, simple_Discriminator as Discriminator

generator_X2Y = Generator()
generator_Y2X = Generator()
discriminator_X = Discriminator()
discriminator_Y = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "CycleGAN_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(X, Y, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
  #  gen_X2Y_output = generator_X2Y(X, training=True)
    D_real_Y_output = discriminator_Y(Y,training = True)
    D_real_X_output = discriminator_X(X,training = True)
    
    fake_Y = generator_X2Y(X,training = True)
    D_fake_Y_output = discriminator_Y(fake_Y,training = True)
    fake_X = generator_Y2X(Y,training = True)
    D_fake_X_output = discriminator_X(fake_X,training = True)
  #  D_fake_loss = discriminator_fake_loss(D_fake_Y_output) + discriminator_fake_loss(D_fake_X_output)
    
    D_loss = discriminator_loss(D_real_X_output, D_fake_X_output) + discriminator_loss(D_real_Y_output, D_fake_Y_output)
    
    disc_params = list(discriminator_X.trainable_variables)+list(discriminator_Y.trainable_variables) 
    discriminator_gradients = disc_tape.gradient(D_loss,disc_params)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc_params))
    
    ## Generator

    fake_Y = generator_X2Y(X,training = True)
    D_fake_Y_output = discriminator_Y(fake_Y,training = True)
    reconstructed_X = generator_Y2X(fake_Y,training = True)
    generator_X_loss, gan_X_loss, l1_X_loss = generator_loss(D_fake_Y_output,fake_Y,X,reconstructed_X)
    
    gen_params = list(generator_X2Y.trainable_variables)+list(generator_Y2X.trainable_variables)
 
    fake_X = generator_Y2X(Y,training = True)
    D_fake_X_output = discriminator_X(fake_X,training = True)
    reconstructed_Y = generator_X2Y(fake_X,training = True)
    generator_Y_loss, gan_Y_loss, l1_Y_loss = generator_loss(D_fake_X_output,fake_X,Y,reconstructed_Y)
    
    gen_total_loss = generator_X_loss+generator_Y_loss
    
    gen_gradients = gen_tape.gradient(gen_total_loss,gen_params)
    generator_optimizer.apply_gradients(zip(gen_gradients,gen_params))
    
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss',gen_total_loss, step = epoch)
        tf.summary.scalar('disc_total_loss',D_loss, step=epoch)




checkpoint_dir = './CycleGAN_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator_X2Y=generator_X2Y,
                                 generator_Y2X=generator_Y2X,
                                 discriminator_X=discriminator_X,
                                 discriminator_Y=discriminator_Y)