from helper import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    data_stream = pd.read_csv('/Users/shivanitomar/Documents/Implementations/shapelet_dictionary/datasets/HAR/data_stream_20k_10subjects.csv')
    batch_size = 2000

    for i in range(0, len(data_stream), batch_size):
        train_batch = data_stream.to_numpy()[i:batch_size]
        train_labels = train_batch[:,-1]
        shapelets = {}
        for label in np.unique(train_labels):
            class_data = train_batch[np.where(train_labels == label)[0]]
            class_labels = train_labels[np.where(train_labels == label)[0]]
            # np.savetxt(f'{label}_train_data_for_shapelets.csv', class_data)
            print(f"Learning shapelets for label - {label}")
            train_data = class_data[:, :-1].astype(np.float64)
            S, A, offsets, F_obj = utils.learn_shapelets_alpha_const(train_data, class_labels, K=10, lambdas=1, r=0.5)
            shapelets[label] = S
            re_old = utils.unsup_obj(train_data, S, A, offsets, lambda_=1) / train_data.shape[0]
            print("reconstruction error using old objective", re_old)
            re, reconstructed_signals = utils.reconstruction_err(train_data, S, A, offsets)
            print("Reconstruction error(MSE) is", re)
            re_2 = utils.compute_mse(train_data, S, A, offsets)
            print("Reconstruction error(MSE) using compute_mse", re_2)

            df_original = pd.DataFrame(list(train_data))
            df_reconstruction = pd.DataFrame(reconstructed_signals)

            for i in range(df_original.shape[0]):
                plt.figure(figsize=(10, 3))
                plt.plot(df_original.iloc[i], label='original', color='blue')
                plt.plot(df_reconstruction.iloc[i], label='reconstruction', color='orange')
                # plt.plot(df_reconstruction_re.iloc[i], label='reconstruction_using_newobj', color='green')
                plt.title(f'Sequence {i}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
        main()