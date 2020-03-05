from matplotlib import pyplot as plt
import os

PATH_output = 'output/'

def generate_images(model, test_input, tar,name):
    prediction = model(test_input, training=True)
    fig,ax = plt.subplots(1,3,figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        ax[i].set_title(title[i])
        ax[i].imshow(display_list[i] * 0.5 + 0.5)
    plt.savefig(name+'.png')
    return 

def generate_multi_images(model, dataset,N,name):
    generator = model.get_generator()
    fig,ax = plt.subplots(N,3,figsize=(15,15*N/2.2))
    for j, (Input, Target) in enumerate(dataset.take(N)):
        prediction = generator(Input, training=True)
        display_list = [Input[0], Target[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
        for i in range(3):
            ax[j][i].set_title(title[i])
            ax[j][i].imshow(display_list[i] * 0.5 + 0.5)
    path = PATH_output + str(model)
    print(str(model))
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/' + name + '.png')
    plt.show()
    return 