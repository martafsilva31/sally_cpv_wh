
"""
sally_training.py

Handles training of SALLY method

Marta Silva (LIP/IST/CERN-ATLAS), 9/11/2023
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging

import sys
import os
from time import strftime
import argparse as ap
import numpy as np

from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ScoreEstimator, Ensemble

# MadMiner output
logging.basicConfig(
  format='%(asctime)-5.5s %(funcName)-20.20s %(levelname)-7.7s %(message)s',
  datefmt='%H:%M',
  level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
  if "madminer" not in key:
    logging.getLogger(key).setLevel(logging.WARNING)


# timestamp for model saving
timestamp = strftime("%d%m%y")

    

def train(input_dir,sample_name,observable_set,nsamples=-1,training_observables='kinematic_only',model_name=''):
    """  


    Parameters:

    input_dir: folder where h5 files and samples are stored

    sample_name: .h5 sample name to augment/train

    observable_set: 'full' (including unobservable degrees of freedom) or 'met' (only observable degrees of freedom)

    training_observables: which observables use to do the training, 'kinematic_only' (for only kinematic observables in full and met observable set), all_observables (kinematic + angular observables in met observable set)

    mode: either create only the augmented data samples (mode=augment_only), only do the SALLY method training (mode=train_only), or augment+training (mode=augment_and_train)

    model_name: model name, given to differentiate between, e.g. different SALLY NN configurations
    """

    # access to the .h5 file with MadMiner settings
    madminer_settings=load_madminer_settings(f'{input_dir}/{observable_set}/{sample_name}.h5',include_nuisance_benchmarks=False)
    
    if nsamples==-1:
        nsamples=madminer_settings[6]
            
    logging.info(f' sample_name: {sample_name}; observable set: {observable_set}; training observables: {training_observables}; nsamples: {nsamples}')



    ########## Training ###########

    # Choose which features to train on 
    # If 'met' or 'full', we use all of them (None), otherwise we select the correct indices
    # for the full observable set we always use all of the variables in the sample
    if observable_set == 'full':
        if training_observables!='kinematic_only':
            logging.warning('for the full observable set, always training with the kinematic_only observable set, which includes all features')
        training_observables='kinematic_only'
        my_features = None
        
    if observable_set == 'parton_level_validation':
        if training_observables == 'kinematic_only':
            my_features = list(range(48))
            my_features = [x for x in my_features if x != 45]
            
    if observable_set == 'parton_level_only_x_pt_w_only':
        if training_observables == 'kinematic_only':
            my_features = list(range(1))
    # else:
    #     if training_observables == 'kinematic_only':
    #         my_features = list(range(48))
    #     if training_observables == 'all_observables':
    #         my_features = None
    #     # removing non-charge-weighted cosDelta (50,51) and charge-weighted cosThetaStar (53)
    #     elif training_observables == 'all_observables_remove_redundant_cos':
    #         my_features = [*range(48),48,49,52,54,55]
    #     elif training_observables == 'ptw_ql_cos_deltaPlus':
    #         my_features = [18,54]      
    #     elif training_observables == 'mttot_ql_cos_deltaPlus':
    #         my_features = [39,54]

    #Create a list of ScoreEstimator objects to add to the ensemble
    estimator = ScoreEstimator(features=my_features, n_hidden=(100,50,),activation="relu") 


    # Run the training of the ensemble
    # result is a list of N tuples, where N is the number of estimators,
    # and each tuple contains two arrays, the first with the training losses, the second with the validation losses
    result = estimator.train(method='sally',                        
        x=f'{input_dir}/{observable_set}/training_samples/alices/x_train_ratio_{sample_name}.npy' ,
        t_xz=f'{input_dir}/{observable_set}/training_samples/alices/t_xz_train_ratio_{sample_name}.npy' ,
        memmap=True,verbose="all",n_workers=4,limit_samplesize=nsamples,n_epochs=50,batch_size=1024,
    )    
   
    # saving ensemble state dict and training and validation losses
    estimator.save(f'{input_dir}/{observable_set}/models/{training_observables}/{model_name}/sally_{sample_name}')
    np.savez(f'{input_dir}/{observable_set}/models/{training_observables}/{model_name}/sally_losses_{sample_name}',result)


if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Creates augmented (unweighted) training samples for the Approximate likelihood with improved cross-entropy estimator method (ALICE). Trains an ensemble of NNs as score estimators.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--observable_set',help="which observable sets to process in one run: full (including unobservable degrees of freedom), met (only observable degrees of freedom), or both sequentially",default=['parton_level_validation'],choices=['full','met','parton_level_validation', "parton_level_only_x_pt_w_only"],nargs="+")

    parser.add_argument('--training_observables',help="observables used for the training: all observables for the full observable set and simple kinematic observables for the met observable set",default='kinematic_only',choices=['kinematic_only','all_observables_remove_redundant_cos'])

    parser.add_argument('--channel',help='lepton+charge flavor channels to augment/train. included to allow parallel training of the different channels',choices=['wph_mu','wph_e','wmh_mu','wmh_e','wmh','wph','wh_mu','wh_e','wh'],nargs="+",default=['wh'])
    
    parser.add_argument('--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds. included to allow sequential training for the different possibilities',choices=['signalOnly_SMonly_noSysts_lhe','signalOnly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe','withBackgrounds_noSysts_lhe'],nargs="+",default='signalOnly_SMonly_noSysts_lhe')

    parser.add_argument('--model_name',help='model name, given to differentiate between, e.g. different ALICE NN configurations',default=timestamp)

    args=parser.parse_args()

    for observable_set in args.observable_set:
        for channel in args.channel:
            for sample_type in args.sample_type:
                
                logging.info(f'observable set: {observable_set}; channel: {channel}; sample type: {sample_type}')
                
                train(input_dir=args.main_dir,observable_set=observable_set,training_observables=args.training_observables,model_name=args.model_name,sample_name=f'{channel}_{sample_type}')

