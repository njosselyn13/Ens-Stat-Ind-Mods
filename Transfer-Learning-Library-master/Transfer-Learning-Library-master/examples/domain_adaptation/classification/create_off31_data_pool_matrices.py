import pandas as pd
import os


train_or_test = 'trainF1' # test

parent_dir = '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/'

adapt_tasks = ['Office31_a2d', 'Office31_a2w', 'Office31_d2w', 'Office31_d2a', 'Office31_w2a', 'Office31_w2d']

folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

model_names = ['afn_subset_exps', 'cdan_subset_exps', 'dann_subset_exps', 'jan_subset_exps', 'mcc_subset_exps']

# loop through all csvs for one TASK across all 5 models, drop image name and leading column
# for each model, create a new sheet
# get GT from the first one, drop the others
# for each model sheet, append the pred and conf score cols horizontally for each fold

for task in adapt_tasks:
    model_sheet_dict = {}
    for mod in model_names:
        fold_dict = {}
        for fold in folds:
            pth1 = parent_dir + 'logs/' + mod + '/Office31/tuned/' + task + '/' + fold + '/' + 'NEW2_off31_' + train_or_test + '_set_gt_pred_conf.csv'
            print(pth1)
            print()
            fold_dict[fold] = pth1

        sheet_df = pd.DataFrame()
        for f in folds:
            pth = fold_dict[f]
            csv_file = pd.read_csv(pth)
            gt_list = csv_file['Ignore_Ground Truth'].dropna().tolist()
            pred_list = csv_file['Ignore_Predicted'].dropna().tolist()
            conf_list = csv_file['outputs'].dropna().tolist()
            conf_lists_list = csv_file['outputs_list'].dropna().tolist()

            if f == 'fold1':
                sheet_df['GT Number'] = gt_list

            sheet_df[mod + '_' + f] = pred_list
            sheet_df[mod + '_' + f + '_conf'] = conf_list

        model_sheet_dict[mod] = sheet_df

    # Specify the output Excel file name
    output_dir = parent_dir + 'off31_eval_models/' + task
    output_file = os.path.join(output_dir, "data_pool_matrix.xlsx")

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write each DataFrame to a separate sheet in the Excel file
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for sheet_name, df in model_sheet_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


