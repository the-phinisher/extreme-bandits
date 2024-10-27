import numpy as np
from xp_helpers import multiprocess_MC_Xtreme

path = "xp"
m = 100

params = {'ExtremeHunter': {'steps': 1000, 'D': 1e-4, 'E': 1e-4, 'b': 1},
          'Threshold_Ascent': {'s': 100, 'delta': 0.1},}

xp_list = [{'coef': [[2.1, 1], [2.3, 1], [1.3, 1], [1.1, 1], [1.9, 1]]}]
names_xp = ['xp'+str(i+1)+'_' for i in range(len(xp_list))]
algs = ['ExtremeHunter', 'Threshold_Ascent']
T_list = [15000]

if __name__ == '__main__':
    for i, xp in enumerate(xp_list):
        for T in T_list:
            print(T)
            batch_size, sample_size = int(np.log(T)**2) + 1, int(np.log(T)) + 1
            args = (xp['coef'], algs, m, T, params)
            res = multiprocess_MC_Xtreme(args, pickle_path=path, caption=names_xp[i]+str(T))
        print('_________________________________________________________________')
        
    import pickle
    import matplotlib.pyplot as plt
    
    file_path = 'xp/xp1_15000.pkl'

    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Display the loaded data
    plt.plot(data['ExtremeHunter']['Best Arm pulls (averaged)'], label="Extreme Hunter")
    plt.plot(data['Threshold_Ascent']['Best Arm pulls (averaged)'], label="Threshold Ascent")
    plt.legend()
    plt.show()
