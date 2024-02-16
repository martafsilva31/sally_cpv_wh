"""
compute_alices_distribution.py

Handles extraction of joint score + likelihood ratio for the test samples and computes the liklelihood ratio with ALICES 

Marta Silva (LIP/IST/CERN-ATLAS), 24/11/2023

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
from madminer.ml import ParameterizedRatioEstimator, Ensemble

# MadMiner output
logging.basicConfig(
  format='%(asctime)-5.5s %(funcName)-20.20s %(levelname)-7.7s %(message)s',
  datefmt='%H:%M',
  level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
  if "madminer" not in key:
    logging.getLogger(key).setLevel(logging.WARNING)


# timestamp for model saving
timestamp = strftime("%d%m%y")


def augment_test(input_dir,sample_name,observable_set,nsamples=100):

  """  
  Extracts the joint likelihood ratio and the joint score for the test partition and plots the likelihood ratio as a function of the parameter theta 

  Parameters:

  input_dir: folder where h5 files and samples are stored

  sample_name: .h5 sample name to augment/train

  observable_set: 'full' (including unobservable degrees of freedom) or 'met' (only observable degrees of freedom)

  """

  # access to the .h5 file with MadMiner settings
  madminer_settings=load_madminer_settings(f'{input_dir}/{observable_set}/{sample_name}.h5',include_nuisance_benchmarks=False)

  if nsamples==-1:
    nsamples=madminer_settings[6]

  logging.info(f'sample_name: {sample_name}; observable set: {observable_set}; nsamples: {nsamples}')
  
  ########## Sample Augmentation ###########
  # # object to create the augmented training samples
  sampler=SampleAugmenter(f'{input_dir}/{observable_set}/{sample_name}.h5')
  
  # Creates a set of testing data (as many as the number of estimators) - centered around the SM
  
  x, theta0, theta1, y, r_xz, t_xz, n_effective = sampler.sample_train_ratio(
  theta0=sampling.random_morphing_points(100, [("gaussian", 0.0, 0.5)]),
  theta1=sampling.benchmark("sm"),
  n_samples=int(nsamples),
  folder=f'{input_dir}/{observable_set}/testing_samples/alices',
  filename=f'test_ratio_{sample_name}',
  sample_only_from_closest_benchmark=True,
  return_individual_n_effective=True,
  partition = "test",
  n_processes = 4
  )
  
  logging.info(f'effective number of samples: {n_effective}')

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Creates augmented (unweighted) training samples for the Approximate likelihood with improved cross-entropy estimator and score method (ALICES). Trains an ensemble of NNs as score estimators.',formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--main_dir',help='folder where to keep everything for MadMiner WH studies, on which we store Madgraph samples and all .h5 files (setup, analyzed events, ...)',required=True)

    parser.add_argument('--observable_set',help="which observable sets to process in one run: full (including unobservable degrees of freedom), met (only observable degrees of freedom), or both sequentially",default=['full','met'],choices=['full','met'],nargs="+")

    parser.add_argument('--nsamples',help="number of events in augmented data sample/number of events on which to train on. Note: if running augmentation and training in separate jobs, these can be different, although number of events in training <= number of events in augmented data sample",type=int,default=-1)

    parser.add_argument('--channel',help='lepton+charge flavor channels to augment/train. included to allow parallel training of the different channels',choices=['wph_mu','wph_e','wmh_mu','wmh_e','wmh','wph','wh_mu','wh_e','wh'],nargs="+",default=['wh_mu','wh_e'])
    
    parser.add_argument('--sample_type',help='sample types to process, without/with samples generated at the BSM benchmark and without/with backgrounds. included to allow sequential training for the different possibilities',choices=['signalOnly_SMonly_noSysts_lhe','signalOnly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe','withBackgrounds_noSysts_lhe'],nargs="+",default=['signalOnly_SMonly_noSysts_lhe','withBackgrounds_SMonly_noSysts_lhe'])

    args=parser.parse_args()

    for observable_set in args.observable_set:
        for channel in args.channel:
            for sample_type in args.sample_type:
                
                logging.info(f'observable set: {observable_set}; channel: {channel}; sample type: {sample_type}')
                
                augment_test(input_dir=args.main_dir,observable_set=observable_set,sample_name=f'{channel}_{sample_type}',nsamples=args.nsamples)
