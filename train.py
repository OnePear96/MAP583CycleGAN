from models.CycleGAN import CycleGAN
from models.GAN import GAN
from models.LGAN import LGAN
from models.LCycleGAN import LCycleGAN
from models.UNN import UNN

Model = 'UNN'

def get_trainer(model_type):
    if (model_type == 'GAN'):
        return GAN, False
    if (model_type == 'LGAN'):
        return LGAN, False
    if (model_type == 'UNN'):
        return UNN, False
    if (model_type == 'CycleGAN'):
        return CycleGAN, True
    if (model_type == 'LCycleGAN'):
        return LCycleGAN, True
    return None


def fit(train_ds, epochs, test_ds, model_type):
  for epoch in range(epochs):
    start = time.time()
    
    '''
    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)
    '''

    Trainer, is_cycle = get_trainer(Model)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      Trainer.train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      Trainer.checkpoint.save(file_prefix = Trainer.checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  Trainer.checkpoint.save(file_prefix = Trainer.checkpoint_prefix)