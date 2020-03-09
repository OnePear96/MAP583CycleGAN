import tensorflow as tf

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss



###TO EDIT

LAMBDA = 100
ALPHA = 50

###

def LCycle_generator_loss(disc_generated_output, gen_output, target, Input, reconstructed_input):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    reconstructed_loss = tf.reduce_mean(tf.abs(Input - reconstructed_input))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss) + (ALPHA * reconstructed_loss)

    return total_gen_loss, gan_loss, l1_loss, reconstructed_loss


def Cycle_generator_loss(disc_generated_output, gen_output, Input, reconstructed_input):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    reconstructed_loss = tf.reduce_mean(tf.abs(Input - reconstructed_input))

    total_gen_loss = gan_loss + (ALPHA * reconstructed_loss) 

    return total_gen_loss , gan_loss, reconstructed_loss


def GAN_generator_loss(disc_generated_output, gen_output):

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    return gan_loss


def LGAN_generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def L_generator_loss(gen_output, target):

  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  
  return l1_loss