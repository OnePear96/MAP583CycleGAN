import datetime
import tensorflow as tf
from tools.losses import L_generator_loss as generator_loss
from basic_models.generator import Generator


generator = Generator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "UNN_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape:
    gen_output = generator(input_image, training=True)
    gen_loss = generator_loss(gen_output, target)

  generator_gradients = gen_tape.gradient(gen_loss,
                                          generator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))


  with summary_writer.as_default():
    tf.summary.scalar('gen_loss', gen_loss, step=epoch)

checkpoint_dir = './UNN_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator)