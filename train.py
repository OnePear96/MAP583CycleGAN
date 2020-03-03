from models.CycleGAN import train_step as CycleGAN, checkpoint as cGANcp, generator_X2Y, generator_Y2X 
from models.GAN import train_step as GAN, checkpoint as GANcp, generator
from models.LGAN import train_step as LGAN, checkpoint as LGANcp, generator
from models.LCycleGAN import train_step as LCycleGAN, checkpoint as LcGANcp, generator_X2Y, generator_Y2X 
from models.UNN import train_step as UNN, checkpoint as Lcp

def get_trainer(model_type):
    if (model_type == 'GAN'):
        return GAN
    if (model_type == 'LGAN'):
        return LGAN
    if (model_type == 'UNN'):
        return UNN
    if (model_type == 'CycleGAN'):
        return CycleGAN
    if (model_type == 'LCycleGAN'):
        return LCycleGAN
    return None


def get_checkpoint(model_type):
    if (model_type == 'GAN'):
        return GANcp
    if (model_type == 'LGAN'):
        return LGANcp
    if (model_type == 'UNN'):
        return Lcp
    if (model_type == 'CycleGAN'):
        return cGANcp
    if (model_type == 'LCycleGAN'):
        return LcGANcp
    return None

def fit(train_ds, epochs, test_ds, model_type):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)