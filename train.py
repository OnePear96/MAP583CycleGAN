from models.CycleGAN import CycleGAN
from models.GAN import GAN
from models.LGAN import LGAN
from models.LCycleGAN import LCycleGAN
from models.Unet import Unet
from tools.data_loader import load_image_s,load_image_u
from tools.output import generate_multi_images
import time
import os

Model = 'Unet'
EPOCHS = 50

def get_trainer(model_type):
    if (model_type == 'GAN'):
        return GAN(), False
    if (model_type == 'LGAN'):
        return LGAN(), False
    if (model_type == 'Unet'):
        return Unet(), False
    if (model_type == 'CycleGAN'):
        return CycleGAN(), True
    if (model_type == 'LCycleGAN'):
        return LCycleGAN(), True
    return None


def fit(train_ds, test_ds, epochs, model_type):
  for epoch in range(epochs):
    start = time.time()

    '''
    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)
    '''

    Trainer, is_cycle = get_trainer(model_type)

    generate_multi_images(Trainer, test_ds,6,epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      Trainer.train_step(input_image, target, epoch)
    print()

    if not os.path.exists(Trainer.checkpoint_prefix):
      os.makedirs(Trainer.checkpoint_prefix)

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      Trainer.checkpoint.save(file_prefix = Trainer.checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  Trainer.checkpoint.save(file_prefix = Trainer.checkpoint_prefix)


if __name__ == "__main__":
    dataloader = load_image_s()
    train_dataset = dataloader.get_train_set()
    test_dataset = dataloader.get_test_set()
    fit(train_dataset, test_dataset, EPOCHS, Model)
