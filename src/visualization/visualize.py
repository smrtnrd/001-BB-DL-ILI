import seaborn as sns
import matplotlib.pyplot as plt

def plot_model_history2(my_first_nn_fitted):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(len(my_first_nn_fitted.history['loss'])), my_first_nn_fitted.history['loss'],linestyle='-', color='blue',label='Training', lw=2)
    ax1.plot(range(len(np.array(test_over_time)[:,0])), np.array(test_over_time)[:,0], linestyle='-', color='green',label='Test', lw=2)
    ax2.plot(range(len(my_first_nn_fitted.history['acc'])), my_first_nn_fitted.history['acc'],linestyle='-', color='blue',label='Training', lw=2)
    ax2.plot(range(len(np.array(test_over_time)[:,1])), np.array(test_over_time)[:,1], linestyle='-', color='green',label='Test', lw=2)
    leg = ax1.legend(bbox_to_anchor=(0.7, 0.9), loc=2, borderaxespad=0.,fontsize=13)
    ax1.set_xticklabels('')
    #ax1.set_yscale('log')
    ax2.set_xlabel('# Epochs',fontsize=14)
    ax1.set_ylabel('Loss',fontsize=14)
    ax2.set_ylabel('Accuracy',fontsize=14)
    plt.show()