import datetime
import tensorflow as tf
from tools.losses import Cycle_generator_loss as generator_loss, discriminator_loss
from basic_models.generator import Generator, simple_Discriminator as Discriminator

generator_X2Y = Generator()
generator_Y2X = Generator()
discriminator_X = Discriminator()
discriminator_Y = Discriminator()
