import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from dev_tools.my_tools import my_makedirs
import glob
import matplotlib.image as mi
import pdb

def evaluation_plot(criteria, label, save_name,csv_file, val=True):
#     pdb.set_trace()
    df = pd.read_csv(csv_file)
    dict_criteria = {}
    dict_criteria['ET'] = [x for x in df[criteria+'_ET'] if not np.isnan(x)][:-5]
    dict_criteria['TC'] = [x for x in df[criteria+'_TC'] if not np.isnan(x)][:-5]
    dict_criteria['WT'] = [x for x in df[criteria+'_WT'] if not np.isnan(x)][:-5]
    plt.figure()

    plt.boxplot(dict_criteria.values(),
                labels=[key+'\nmean: %.2f'%(np.mean(dict_criteria[key])) for key in dict_criteria.keys()])
    plt.ylabel(label)
    dataset_type = 'Val' if val else 'Training'
    plt.title(label + ' Boxplot of ' + dataset_type + ' Dataset')
    plt.savefig(save_name,dpi=600)
    
def draw_evaluate(csv_file,save_dir,val=True, fig_format='png'):
    my_makedirs(save_dir)
    evaluation_plot('Dice', 'Dice Coefficient', os.path.join(save_dir,'dice_val.'+fig_format), csv_file, val=val)
    evaluation_plot('Sensitivity', 'Sensitivity', os.path.join(save_dir,'sensitivity_val.'+fig_format),csv_file, val=val)
    evaluation_plot('Specificity', 'Specificity', os.path.join(save_dir,'specificity_val.'+fig_format),csv_file, val=val)
    evaluation_plot('Hausdorff95', 'Hausdorff Disdance', os.path.join(save_dir,'hausdorff_val.'+fig_format),csv_file, val=val)
    
def four_in_all(png_fold, fig_format='pdf'):
    files = glob.glob(os.path.join(png_fold,'*'))
    FONTSIZE=20
    plt.figure()
    fig, axs = plt.subplots(2, 2,figsize=(15,15))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.0, hspace=0.0)

    img = mi.imread(files[0])
    axs[0,0].imshow(img)
    axs[0,0].axis('off')
    
    img = mi.imread(files[1])
    axs[0,1].imshow(img)
    axs[0,1].axis('off')
    
    img = mi.imread(files[2])
    axs[1,0].imshow(img)
    axs[1,0].axis('off')
    
    img = mi.imread(files[3])
    axs[1,1].imshow(img)
    axs[1,1].axis('off')
    
    fig.savefig(os.path.join(png_fold,'four_in_all.'+fig_format),dpi=400)
    return
