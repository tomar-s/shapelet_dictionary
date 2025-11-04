import numpy as np
import pandas as pd

if __name__ == "__main__":


# # For autotuned results
#     data = {
#         "Dataset": ["Monash Traffic", "Monash Weather", "ETTh", "Ercot", "Australian Electricity",
#                     "Exchange rate", "Monash_Fred_md", "NN5", "M5", "ETTm"],
#         "Seed_42": [0.777, 0.813, 0.838, 0.649, 0.836, 1.682, 0.518, 0.622, 0.925, 0.718],
#         "Seed_56": [0.735, 0.823, 0.790, 0.516, 0.720, 1.859, 0.519, 0.608, 0.925, 0.786],
#         "Seed_101": [0.742, 0.825, 0.768, 0.658, 0.763, 1.346, 0.496, 0.625, 0.926, 0.692],
#         "Seed_561": [0.739, 0.822, 0.801, 0.602, 0.892, 1.540, 0.510, 0.602, 0.926, 0.685],
#         "Seed_8": [0.738, 0.824, 0.786, 0.608, 0.947, 1.732, 0.511, 0.642, 0.926, 0.686]
#     }
#
#     df = pd.DataFrame(data)
#
#     # Calculate the mean and standard deviation across the 5 seeds for each dataset
#     df["Mean"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].mean(axis=1)
#     df["StdDev"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].std(axis=1)
#
#     # Display the DataFrame with Mean and StdDev
#     print(df[["Dataset", "Mean", "StdDev"]])

# For BOHB results
    data = {
        "Dataset": ["Monash Traffic", "Monash Weather", "ETTh", "Ercot", "Australian Electricity",
                    "Exchange rate", "Monash_Fred_md", "NN5", "m5", "ETTm"],
        "Seed_42": [0.767, 0.816 , 0.807, 0.582, 0.642, 3.107, 0.482, 0.593, 0.923, 0.739 ],
        "Seed_56": [0.773, 0.820, 0.780, 0.653, 0.752, 1.991, 0.503, 0.603, 0.922, 0.651 ],
        "Seed_101": [0.756, 0.814, 0.768, 0.612, 0.738, 1.679, 0.563, 0.614, 0.924, 0.676 ],
        "Seed_561": [0.761, 0.824, 0.795, 0.650, 0.700, 2.694, 0.513, 0.582, 0.923, 0.695 ],
        "Seed_8": [0.768, 0.815, 0.835,0.587, 0.855, 1.835, 0.638, 0.611, 0.922, 0.708 ],


    }

    df = pd.DataFrame(data)

    # Calculate the mean and standard deviation across the 5 seeds for each dataset
    df["Mean"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].mean(axis=1)
    df["StdDev"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].std(axis=1)

    # Display the DataFrame with Mean and StdDev
    print(df[["Dataset", "Mean", "StdDev"]])

# # For autotuned results (hyperopt)
#     data = {
#         "Dataset": ["Monash Traffic", "Monash Weather",  "ETTh", "Ercot", "Australian Electricity",
#                     "Exchange rate", "Monash_Fred_md", "NN5"],
#         "Seed_42": [0.873, 0.826, 0.795, 0.671, 0.811, 1.575, 0.547, 0.588],
#         "Seed_56": [0.763, 0.828, 0.793, 0.615, 0.790, 2.702, 0.542, 0.607],
#         "Seed_101": [0.760, 0.820, 0.797, 0.618, 0.832, 2.637,0.566, 0.583],
#         "Seed_561": [0.767, 0.822, 0.787, 0.552, 0.847, 1.838, 0.600, 0.604],
#         "Seed_8": [0.728, 0.822, 0.795, 0.550, 0.863, 1.967, 0.513, 0.599]
#     }
#
#     df = pd.DataFrame(data)
#
#     # Calculate the mean and standard deviation across the 5 seeds for each dataset
#     df["Mean"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].mean(axis=1)
#     df["StdDev"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].std(axis=1)
#
#     # Display the DataFrame with Mean and StdDev
#     print(df[["Dataset", "Mean", "StdDev"]])

    # # For full fine-tuning results
    # data = {
    #     "Dataset": ["Monash Traffic", "Monash Weather", "ETTh", "Ercot", "Australian Electricity",
    #                 "Exchange rate", "Monash_Fred_md", "NN5", "m5", "ETTm"],
    #     "Seed_42": [0.726, 0.814, 0.789, 0.547, 1.023, 1.903, 0.534, 0.603, 0.934, 0.770],
    #     "Seed_56": [0.725, 0.813, 0.763, 0.635, 0.906, 2.024, 0.518, 0.603, 0.9346320617569972, 0.785],
    #     "Seed_101": [0.739, 0.822, 0.773, 0.580, 0.832, 1.777, 0.499, 0.602, 0.9341974772327863, 0.799],
    #     "Seed_561": [0.723, 0.820, 0.807, 0.618, 0.884, 1.797, 0.504, 0.608, 0.9349918781812064, 0.756],
    #     "Seed_8": [0.726, 0.823, 0.784, 0.615, 0.991, 1.730, 0.499, 0.599, 0.9345510523357763, 0.776],
    #
    # }
    #
    # df = pd.DataFrame(data)
    #
    # # Calculate the mean and standard deviation across the 5 seeds for each dataset
    # df["Mean"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].mean(axis=1)
    # df["StdDev"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].std(axis=1)
    #
    # # Display the DataFrame with Mean and StdDev
    # print(df[["Dataset", "Mean", "StdDev"]])

    # # For default LoRA results
    # data = {
    #     "Dataset": ["Monash Traffic", "Monash Weather", "ETTh", "Ercot", "Australian Electricity",
    #                 "Exchange rate", "Monash_Fred_md", "NN5", "ETTm"],
    #     "Seed_42": [0.775, 0.847, 0.804, 0.527, 1.146, 1.909, 0.513, 0.603, 0.684],
    #     "Seed_56": [0.774, 0.843, 0.833, 0.585, 1.191, 1.828, 0.521, 0.604, 0.721],
    #     "Seed_101": [0.777, 0.845, 0.823, 0.557, 1.144, 1.789, 0.491, 0.605, 0.706],
    #     "Seed_561": [0.775, 0.853, 0.860, 0.621, 1.168, 1.926, 0.511, 0.606, 0.736],
    #     "Seed_8": [0.778, 0.847 , 0.829, 0.537, 1.101, 1.901, 0.503, 0.605, 0.708],
    # }
    #
    # df = pd.DataFrame(data)
    #
    # # Calculate the mean and standard deviation across the 5 seeds for each dataset
    # df["Mean"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8"]].mean(axis=1)
    # df["StdDev"] = df[["Seed_42", "Seed_56", "Seed_101", "Seed_561", "Seed_8" ]].std(axis=1)
    #
    # # Display the DataFrame with Mean and StdDev
    # print(df[["Dataset", "Mean", "StdDev"]])

    # import numpy as np
    #
    #
    # def calculate_mase_improvement(zero_shot_mase, autotune_mase):
    #     """
    #     Calculate the percentage improvement in MASE between zero-shot and AutoTune approaches.
    #
    #     :param zero_shot_mase: List of MASE scores for zero-shot approach
    #     :param autotune_mase: List of MASE scores for AutoTune approach
    #     :return: Average percentage improvement
    #     """
    #     if len(zero_shot_mase) != len(autotune_mase):
    #         raise ValueError("The number of MASE scores must be the same for both approaches")
    #
    #     improvements = []
    #     for zs, at in zip(zero_shot_mase, autotune_mase):
    #         improvement = (zs - at) / zs * 100
    #         improvements.append(improvement)
    #
    #     average_improvement = np.mean(improvements)
    #     return average_improvement, improvements
    #
    # zero_shot_mase = [0.853, 0.859, 0.795, 0.582, 0.965, 2.054, 0.473, 0.648, 0.942, 0.709]
    # autotune_mase = [0.746, 0.821, 0.796, 0.565, 0.831, 1.631, 0.510, 0.619, 0.925, 0.713]
    #
    # # zero_shot_ood = [2.054, 0.648, 0.942]
    # # autotune_mase_ood = [1.631, 0.619, 0.925]
    #
    # average_improvement, improvements  = calculate_mase_improvement(zero_shot_mase, autotune_mase)
    # print(f"Average improvement in MASE: {average_improvement:.2f}%")
    # print(improvements)

    # rmse_zero_shot = 0.6707 # Replace with actual zero-shot RMSE value
    # rmse_fine_tuning = 0.630  # Replace with actual fine-tuning RMSE value
    #
    # # Calculate percentage improvement
    # percentage_improvement = ((rmse_zero_shot - rmse_fine_tuning) / rmse_zero_shot) * 100
    #
    # print(f"Percentage Improvement: {percentage_improvement:.2f}%")


    # # average improvement based on 5 runs of TTM on MORE Data....FLs (24, 48, 60, 96, 192)
    # zero_shot_mse = [0.3362594544887543, 0.3534010648727417, 0.3582810163497925, 0.3651297092437744, 0.38274073600769043]
    # autotune_mse = [0.13808107376098633, 0.1329152137041092, 0.1454494148492813, 0.16120068728923798, 0.11704809218645096]
    # finetune_mse = [0.12083568423986400, 0.1486613005399700, 0.16236844658851600, 0.16617931425571400, 0.13459020853042600]
    #
    # # # average improvement based on 5 runs of TTM on ETTh1 Data....FLs (24, 48, 60, 96, 192)
    # # zero_shot_mse = [0.4652969539165500, 0.5081443190574646, 0.5238552689552307, 0.5513477921485901, 0.6000331044197083]
    # # autotune_mse = [0.4403444230556490, 0.49569955468177795, 0.5134906768798828, 0.5473090410232544, 0.6025950908660889]
    # # finetune_mse = [0.4490542411804200, 0.49634623527526855, 0.513807475566864, 0.5499274134635925,0.6203089356422424]
    #
    # # # average improvement based on 5 runs of TTM on weather Data....FLs (24, 48, 60, 96, 192)
    # # zero_shot_mse = [0.0898508429527283, 0.11608617007732400, 0.12625479698181200, 0.15046174824237800, 0.19524487853050200 ]
    # # autotune_mse = [0.08721335232257840, 0.11067628860473600, 0.11965091526508300, 0.14223948121070900, 0.18973055481910700 ]
    # # finetune_mse = [0.08827001601457600, 0.11286827176809311, 0.12251710891723633, 0.14862345159053802, 0.1965937465429306]
    #
    # # # average improvement based on 5 runs of TTM on ETTh2 Data....FLs (24, 48, 60, 96, 192)
    # # zero_shot_mse = [0.16964241862297100, 0.2189026027917860, 0.23693478107452400, 0.2757423520088200, 0.3456672430038450]
    # # autotune_mse = [0.15923723578453100, 0.21316611766815200, 0.23434020578861200, 0.26626506447792100, 0.3477340042591100]
    # # finetune_mse = [0.15931913256645203, 0.21038593351840973, 0.2290443629026413, 0.2744038999080658, 0.3639475405216217]
    #
    # mean_zero_shot = np.mean(zero_shot_mse)
    # std_zero_shot = np.std(zero_shot_mse)
    # mean_autotune = np.mean(autotune_mse)
    # std_autotune = np.std(autotune_mse)
    # mean_finetune = np.mean(finetune_mse)
    # std_finetune = np.std(finetune_mse)
    # print(f"The mean_zero_shot and std is: {mean_zero_shot} {std_zero_shot}")
    # print(f"The mean_autotune and std is: {mean_autotune} {std_autotune}")
    # print(f"The mean_finetune and std is: {mean_finetune} {std_finetune}")
    # # 0.5142942070960999
    #
    # # Calculate percentage improvement
    # TTM_more_percentage_improvement = ((mean_zero_shot - mean_autotune) / mean_zero_shot) * 100
    # print(f"Percentage Improvement: {TTM_more_percentage_improvement:.2f}%")
    #
    # # Calculate percentage improvement
    # TTM_more_percentage_improvement_finetune = ((mean_finetune - mean_autotune) / mean_finetune) * 100
    # print(f"Percentage Improvement: {TTM_more_percentage_improvement_finetune:.2f}%")
    #
    # TTM_more_percentage_improvement_FL_192 = ((0.13459020853042600 - 0.11704809218645096) / 0.13459020853042600) * 100
    # print(f"Percentage Improvement: {TTM_more_percentage_improvement_FL_192:.2f}%")
