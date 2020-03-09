from models.CycleGAN import CycleGAN
from models.GAN import GAN
from models.LGAN import LGAN
from models.LCycleGAN import LCycleGAN
from models.Unet import Unet
from tools.data_loader import load_image_s,load_image_u
from tools.output import generate_multi_images
from tools.args import create_parser
import tensorflow as tf
import time
import os



def get_trainer(model_name):
    if (model_name == 'gan'):
        return GAN(), False
    if (model_name == 'lgan'):
        return LGAN(), False
    if (model_name == 'unet'):
        return Unet(), False
    if (model_name == 'cyclegan'):
        return CycleGAN(), True
    if (model_name == 'lcyclegan'):
        return LCycleGAN(), True
    if (model_name == 'cyclegan_inria'):
        return CycleGAN(cX=3,cY=1), True
    return None



def fit(train_ds, test_ds,epochs,model_name, restore = False, Supervised = True, train_dsB = None, test_dsB = None):
  Trainer, is_cycle = get_trainer(model_name)
  if restore:
    Trainer.checkpoint.restore(tf.train.latest_checkpoint(Trainer.checkpoint_dir))
  for epoch in range(epochs):
    start = time.time()

    if not Supervised:
      generate_multi_images(Trainer, test_ds,6,epoch,cycle = is_cycle, Supervised = False, datasetB = test_dsB)
      print("Epoch: ", epoch)
      # Train
      n=0
      for (X,Y) in zip(train_ds,train_dsB):
        print('.', end='')
        if (n+1)%100 == 0:
          print()
        n +=1
        Trainer.train_step(X,Y,epoch)
      
    else:
      generate_multi_images(Trainer, test_ds,6,epoch,cycle = is_cycle)
      print("Epoch: ", epoch)
      # Train
      for n, (input_image, target) in enumerate(train_ds):
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

    opts = create_parser()
    model_name = opts.model
    epoch = opts.epochs
    restore = opts.load

    if opts.dataset == 'inria':
      dataloader = load_image_u(path = './data/Inria/', inria = True,name_X = 'A', name_Y = 'B')
      train_dataset_X,train_dataset_Y = dataloader.get_train_set()
      test_dataset_X = dataloader.get_test_set()
      model_name = 'cyclegan_inria'
      fit(train_dataset_X, test_dataset_X, epoch,model_name,restore,Supervised=False, train_dsB = train_dataset_Y)
    else:
      dataloader = load_image_s()
      train_dataset = dataloader.get_train_set()
      test_dataset = dataloader.get_test_set()
      fit(train_dataset, test_dataset, epoch,model_name,restore)
