from matplotlib import pyplot as plt
import os

PATH_output = 'output/'

def generate_images(model, test_input, tar,name, cycle = False):
    if cycle:
        generator_X2Y = model.generator_X2Y
        generator_Y2X = model.generator_Y2X
        pred_Y = generator_X2Y(test_input, training = True)
        pred_X = generator_Y2X(tar, training = True)
        fig,ax = plt.subplots(1,4,figsize=(15,15))
        display_list = [test_input[0], tar[0], pred_X[0], pred_Y[0]]
        title = ['X', 'Y', 'Predicted X', 'Predicted Y']
        for i in range(4):
            ax[i].set_title(title[i])
            ax[i].imshow(display_list[i] * 0.5 + 0.5)
        plt.savefig(name+'.png')
        return
    else:
        prediction = model(test_input, training=True)
        fig,ax = plt.subplots(1,3,figsize=(15,15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            ax[i].set_title(title[i])
            try:
                ax[i].imshow(display_list[i] * 0.5 + 0.5)
            except:
                ax[i].imshow(display_list[i][:,:,0] * 0.5 + 0.5)
        plt.savefig(name+'.png')
        return 

def generate_multi_images(model, dataset,N,name,cycle = False,datasetB=None, Supervised = True):
    path = PATH_output + str(model)
    if Supervised==False:
        if cycle:
            if datasetB is None:
                generator = model.generator_X2Y
                fig,ax = plt.subplots(N,2,figsize=(15,15*N/2.2))
                for j, Input in enumerate(dataset.take(N)):
                    pred = generator(Input, training = True)
                    display_list = [Input[0], pred[0]]
                    title = ['input', 'prediction']
                    for i in range(2):
                        ax[j][i].axis('off')
                        ax[j][i].set_title(title[i])
                        try:
                            ax[j][i].imshow(display_list[i] * 0.5 + 0.5)
                        except:
                            ax[j][i].imshow(display_list[i][:,:,0] * 0.5 + 0.5)
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path + '/' + str(name) + '.png')
                return
            else:    
                generator_X2Y = model.generator_X2Y
                generator_Y2X = model.generator_Y2X
                fig,ax = plt.subplots(N,4,figsize=(15,15*N/2.2))
                ax.axis('off')
                j = 0
                for (Input, Target) in zip(dataset.take(N),datasetB.take(N)):
                    pred_Y = generator_X2Y(Input, training = True)
                    pred_X = generator_Y2X(Target, training = True)
                    display_list = [Input[0], Target[0], pred_X[0], pred_Y[0]]
                    title = ['X', 'Y', 'Predicted X', 'Predicted Y']
                    for i in range(4):
                        ax[j][i].axis('off')
                        ax[j][i].set_title(title[i])
                        ax[j][i].imshow(display_list[i] * 0.5 + 0.5)
                    j += 1
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path + '/' + str(name) + '.png')
                return
        else:
            generator = model.generator
            fig,ax = plt.subplots(N,2,figsize=(15,15*N/2.2))
            plt.axis('off')
            for j, Input in enumerate(dataset.take(N)):
                pred = generator(Input)
                display_list = [Input[0], pred[0]]
                title = ['X', 'Predicted Y']
                for i in range(2):
                    ax[j][i].axis('off')
                    ax[j][i].set_title(title[i])
                    ax[j][i].imshow(display_list[i] * 0.5 + 0.5)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + '/' + str(name) + '.png')
            return

    if cycle:
        generator_X2Y = model.generator_X2Y
        generator_Y2X = model.generator_Y2X
        fig,ax = plt.subplots(N,4,figsize=(15,15*N/2.2))
        plt.axis('off')
        for j, (Input, Target) in enumerate(dataset.take(N)):
            pred_Y = generator_X2Y(Input, training = True)
            pred_X = generator_Y2X(Target, training = True)
            display_list = [Input[0], Target[0], pred_X[0], pred_Y[0]]
            title = ['X', 'Y', 'Predicted X', 'Predicted Y']
            for i in range(4):
                ax[j][i].axis('off')
                ax[j][i].set_title(title[i])
                ax[j][i].imshow(display_list[i] * 0.5 + 0.5)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/' + str(name) + '.png')
        return

    else:
        generator = model.generator
        fig,ax = plt.subplots(N,3,figsize=(15,15*N/2.2))
        plt.axis('off')
        for j, (Input, Target) in enumerate(dataset.take(N)):
            prediction = generator(Input, training=True)
            display_list = [Input[0], Target[0], prediction[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']
            for i in range(3):
                ax[j][i].axis('off')
                ax[j][i].set_title(title[i])
                ax[j][i].imshow(display_list[i] * 0.5 + 0.5)
    #    print(str(model))
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/' + str(name) + '.png')
    #    plt.show()
        return 