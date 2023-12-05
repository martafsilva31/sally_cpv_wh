from __future__ import absolute_import, division, print_function, unicode_literals
import logging

import sys
import os
from time import strftime
import argparse as ap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator, Ensemble

def validation_plot(model_path, args, channel):
    
    alices = ParameterizedRatioEstimator()
    alices.load(model_path)
    
    joint_likelihood_ratio = np.load(f"{args.main_dir}/testing_samples/alices/r_xz_test_ratio_{channel}_{args.sample_type}.npy")
    joint_score = np.load(f"{args.main_dir}/testing_samples/alices/t_xz_test_ratio_{channel}_{args.sample_type}.npy")
    thetas = np.load(f"{args.main_dir}/testing_samples/alices/theta0_test_ratio_{channel}_{args.sample_type}.npy")
    x = np.load(f"{args.main_dir}/testing_samples/alices/x_test_ratio_{channel}_{args.sample_type}.npy")

    joint_likelihood_ratio_log = np.log(joint_likelihood_ratio)
    log_r_hat, t_hat = alices.evaluate_log_likelihood_ratio(x=x, theta = thetas, test_all_combinations=False, evaluate_score = True)
    
    fig=plt.figure()
    
    plt.scatter(-2*log_r_hat,-2*joint_likelihood_ratio_log)
    min_val = min(-2 * log_r_hat.min(), -2 * joint_likelihood_ratio_log.min())
    max_val = max(-2 * log_r_hat.max(), -2 * joint_likelihood_ratio_log.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
    

    plt.xlabel(r'True log likelihood ratio log r(x)')
    plt.ylabel(r'Estimated log likelihood ratio log $\hat{r}(x)$ (ALICES)')
    plt.tight_layout()
    
    fig_score = plt.figure()
    plt.scatter(-2*t_hat,-2*joint_score)
    min_val = min(-2 * t_hat.min(), -2 * joint_score.min())
    max_val = max(-2 * t_hat.max(), -2 * joint_score.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
    plt.xlabel(r'True score t(x)')
    plt.ylabel(r'Estimated score $\hat{t}(x)$ (ALICES)')
    plt.tight_layout()
    
    return fig, fig_score

if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Computes distributions of different variables for signal and backgrounds.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-dir','--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('-pdir','--plot_dir',help='folder where to save plots to',required=True)

    parser.add_argument('-s','--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds.',choices=['signalOnly_SMonly_noSysts_lhe','signalOnly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe','withBackgrounds_noSysts_lhe'],default='withBackgrounds_SMonly_noSysts_lhe')
        
    parser.add_argument('-c','--channel',help='lepton+charge flavor channels to plot.',choices=['wph_mu','wph_e','wmh_mu','wmh_e','wmh','wph','wh_mu','wh_e','wh'],default=['wh'],nargs="+")

    parser.add_argument('-ao','--alices_observables',help='which of the ALICES training input variable set models to use',required='alices' in sys.argv)

    parser.add_argument('-am','--alices_model',help='which of the ALICES models (for each of the input variable configurations) to use.',required='alices' in sys.argv)

    args=parser.parse_args()

    os.makedirs(f'{args.plot_dir}/',exist_ok=True)

    for channel in args.channel:

    
        validation_llr, validation_score = validation_plot(f"{args.main_dir}/models/{args.alices_observables}/{args.alices_model}/alices_{channel}_{args.sample_type}", args, channel)
    
                
        validation_llr.savefig(f'{args.plot_dir}/alices_validation_llr_{channel}_{args.sample_type}_{args.alices_observables}_{args.alices_model}.pdf')
                
        validation_score.savefig(f'{args.plot_dir}/alices_validation_score_{channel}_{args.sample_type}_{args.alices_observables}_{args.alices_model}.pdf')
