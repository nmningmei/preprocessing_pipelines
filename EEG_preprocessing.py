#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:44:28 2021

@author: nmei

The EEG preprocessing pipeline with flexible steps

MNE-python version: 0.21.2

"""

import re
import mne
import scipy

import numpy as np

from scipy.stats import kurtosis
from numpy import nanmean
from mne.utils import logger

def str2int(x):
    if type(x) is str:
        return float(re.findall(r'\d+',x)[0])
    else:
        return x
def find_outliers(X, threshold=3.0, max_iter=2):
    """Find outliers based on iterated Z-scoring.
 
    This procedure compares the absolute z-score against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.
    
    ########ATTENTION ATTENTION ATTENTION#####
    # This function if removed from MNE-python code base

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.
    max_iter : int
        The maximum number of iterations.
 
    Returns
    -------
    bad_idx : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    from scipy.stats import zscore
    my_mask = np.zeros(len(X), dtype=np.bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        this_z = np.abs(zscore(X))
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break
 
    bad_idx = np.where(my_mask)[0]
    return bad_idx
def hurst(x):
    """Estimate Hurst exponent on a timeseries.
    The estimation is based on the second order discrete derivative.
    Parameters
    ----------
    x : 1D numpy array
        The timeseries to estimate the Hurst exponent for.
    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given timeseries.
    """
    y = np.cumsum(np.diff(x, axis=1), axis=1)

    b1 = [1, -2, 1]
    b2 = [1,  0, -2, 0, 1]

    # second order derivative
    y1 = scipy.signal.lfilter(b1, 1, y, axis=1)
    y1 = y1[:, len(b1) - 1:-1]  # first values contain filter artifacts

    # wider second order derivative
    y2 = scipy.signal.lfilter(b2, 1, y, axis=1)
    y2 = y2[:, len(b2) - 1:-1]  # first values contain filter artifacts

    s1 = np.mean(y1 ** 2, axis=1)
    s2 = np.mean(y2 ** 2, axis=1)

    return 0.5 * np.log2(s2 / s1)

def _freqs_power(data, sfreq, freqs):
    fs, ps = scipy.signal.welch(data, sfreq,
                                nperseg=2 ** int(np.log2(10 * sfreq) + 1),
                                noverlap=0,
                                axis=-1)
    return np.sum([ps[..., np.searchsorted(fs, f)] for f in freqs], axis=0)

def faster_bad_channels(epochs, picks=None, thres=3, use_metrics=None):
    """Implements the first step of the FASTER algorithm.
    
    This function attempts to automatically mark bad EEG channels by performing
    outlier detection. It operated on epoched data, to make sure only relevant
    data is analyzed.
    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs for which bad channels need to be marked
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. A channel
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'variance', 'correlation', 'hurst', 'kurtosis', 'line_noise'
        Defaults to all of them.
    Returns
    -------
    bads : list of str
        The names of the bad EEG channels.
    """
    metrics = {
        'variance':    lambda x: np.var(x, axis=1),
        'correlation': lambda x: nanmean(
                           np.ma.masked_array(
                               np.corrcoef(x),
                               np.identity(len(x), dtype=bool)
                           ),
                           axis=0),
        'hurst':       lambda x: hurst(x),
        'kurtosis':    lambda x: kurtosis(x, axis=1),
        'line_noise':  lambda x: _freqs_power(x, epochs.info['sfreq'],
                                              [50, 60]),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
    if use_metrics is None:
        use_metrics = metrics.keys()

    # Concatenate epochs in time
    data = epochs.get_data()
    data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    data = data[picks]

    # Find bad channels
    bads = []
    for m in use_metrics:
        s = metrics[m](data)
        b = [epochs.ch_names[picks[i]] for i in find_outliers(s, thres)]
        logger.info('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()

def _deviation(data):
    """Computes the deviation from mean for each channel in a set of epochs.
    This is not implemented as a lambda function, because the channel means
    should be cached during the computation.
    
    Parameters
    ----------
    data : 3D numpy array
        The epochs (#epochs x #channels x #samples).
    Returns
    -------
    dev : 1D numpy array
        For each epoch, the mean deviation of the channels.
    """
    ch_mean = np.mean(data, axis=2)
    return ch_mean - np.mean(ch_mean, axis=0)

def faster_bad_epochs(epochs, picks=None, thres=3, use_metrics=None):
    """Implements the second step of the FASTER algorithm.
    
    This function attempts to automatically mark bad epochs by performing
    outlier detection.
    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation'
        Defaults to all of them.
    Returns
    -------
    bads : list of int
        The indices of the bad epochs.
    """

    metrics = {
        'amplitude': lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        'deviation': lambda x: np.mean(_deviation(x), axis=1),
        'variance':  lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    data = epochs.get_data()[:, picks, :]

    bads = []
    for m in use_metrics:
        s = metrics[m](data)
        b = find_outliers(s, thres)
        logger.info('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()

def _power_gradient(ica, source_data):
    # Compute power spectrum
    f, Ps = scipy.signal.welch(source_data, ica.info['sfreq'])

    # Limit power spectrum to upper frequencies
    Ps = Ps[:, np.searchsorted(f, 25):np.searchsorted(f, 45)]

    # Compute mean gradients
    return np.mean(np.diff(Ps), axis=1)


def faster_bad_components(ica, epochs, thres=3, use_metrics=None):
    """Implements the third step of the FASTER algorithm.
    
    This function attempts to automatically mark bad ICA components by
    performing outlier detection.
    Parameters
    ----------
    ica : Instance of ICA
        The ICA operator, already fitted to the supplied Epochs object.
    epochs : Instance of Epochs
        The untransformed epochs to analyze.
    thres : float
        The threshold value, in standard deviations, to apply. A component
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'eog_correlation', 'kurtosis', 'power_gradient', 'hurst',
            'median_gradient'
        Defaults to all of them.
    Returns
    -------
    bads : list of int
        The indices of the bad components.
    See also
    --------
    ICA.find_bads_ecg
    ICA.find_bads_eog
    """
    source_data = ica.get_sources(epochs).get_data().transpose(1,0,2)
    source_data = source_data.reshape(source_data.shape[0], -1)

    metrics = {
        'eog_correlation': lambda x: x.find_bads_eog(epochs)[1],
        'kurtosis':        lambda x: kurtosis(
                               np.dot(
                                   x.mixing_matrix_.T,
                                   x.pca_components_[:x.n_components_]),
                               axis=1),
        'power_gradient':  lambda x: _power_gradient(x, source_data),
        'hurst':           lambda x: hurst(source_data),
        'median_gradient': lambda x: np.median(np.abs(np.diff(source_data)),
                                               axis=1),
        'line_noise':  lambda x: _freqs_power(source_data,
                                              epochs.info['sfreq'], [50, 60]),
    }

    if use_metrics is None:
        use_metrics = metrics.keys()

    bads = []
    for m in use_metrics:
        scores = np.atleast_2d(metrics[m](ica))
        for s in scores:
            b = find_outliers(s, thres)
            logger.info('Bad by %s:\n\t%s' % (m, b))
            bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()

def faster_bad_channels_in_epochs(epochs, picks=None, thres=3, use_metrics=None):
    """Implements the fourth step of the FASTER algorithm.
    
    This function attempts to automatically mark bad channels in each epochs by
    performing outlier detection.
    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation', 'median_gradient'
        Defaults to all of them.
    Returns
    -------
    bads : list of lists of int
        For each epoch, the indices of the bad channels.
    """

    metrics = {
        'amplitude':       lambda x: np.ptp(x, axis=2),
        'deviation':       lambda x: _deviation(x),
        'variance':        lambda x: np.var(x, axis=2),
        'median_gradient': lambda x: np.median(np.abs(np.diff(x)), axis=2),
        'line_noise':      lambda x: _freqs_power(x, epochs.info['sfreq'],
                                                  [50, 60]),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    
    data = epochs.get_data()[:, picks, :]

    bads = [[] for i in range(len(epochs))]
    for m in use_metrics:
        s_epochs = metrics[m](data)
        for i, s in enumerate(s_epochs):
            b = [epochs.ch_names[picks[j]] for j in find_outliers(s, thres)]
            logger.info('Epoch %d, Bad by %s:\n\t%s' % (i, m, b))
            bads[i].append(b)

    for i, b in enumerate(bads):
        if len(b) > 0:
            bads[i] = np.unique(np.concatenate(b)).tolist()
            
    return bads

def _check_bad_channels(_epochs,picks,func = faster_bad_channels):
        bad_channels_list = func(_epochs,picks = picks)
        for ch_name in bad_channels_list:
            _epochs.info['bads'].append(ch_name)
        return bad_channels_list,_epochs

class PreprocessingPipeline(object):
    """
    Parameters
    ------
    raw : mne.io.Raw, raw continue EEG recording
    events : numpy.narray, n_events X 3, events matrix
    event_id : dict, event ids that match the last colun of the events matrix
    notch_filter : int or float, default = 50
        band for notch filtering
    perform_ICA : bool, default = True
        whether plan to perform ICA preprocessing on the data
    highpass : int or float, default = None
        if not None, apply high pass filtering on the data
    highpass_ICA : int or float, default = 1.
        highpass a copy of the raw data at 1. Hz for ICA because ICA is sensitive
        to low-frequency drifts. We want to remove them before fitting the ICA
    lowpass : int or float, default = None
        if not None, apply low pass filtering on the data
    tmin : int or float,default = -0.5
        the starting time point for defining the Epochs
    tmax : int or float, default = 2.5
        the ending time point for defining the Epochs
    baseline : tuple of int or float, default = (-0.5,0)
        the baseline for defining the Epochs
    interpolate_bad_channels : bool, default = True
        whether to perform bad channels interpolation
    
    Examples
    -----
    >>> # first we initialize the pipeline as a data container
    >>> pipeline        = PreprocessingPipeline(raw,
    >>>                                         events,
    >>>                                         event_id,
    >>>                                         )
    >>> # now we preprocess the data step by step
    >>> pipeline.re_refernce()
    >>> pipeline.notch_filtering()
    >>> pipeline.filtering()
    >>> pipeline.epoching()
    >>> pipeline.mark_bad_channels()
    >>> pipeline.mark_bad_epochs()
    >>> pipeline.mark_bad_channels_for_each_epoch()
    >>> pipeline.fit_ica()
    >>> pipeline.mark_bad_ica_components_by_FASTER()
    >>> pipeline.detect_artifacts()
    >>> pipeline.apply_ica()
    >>> pipeline.final_step()
    >>> clean_epochs = piplein.clean_epochs
    >>> for event_name,val in event_id.items():
    >>>     evoked = clean_epochs[event_name].average()
    >>>     evoked.plot_joint(title = event_name)
    """
    def __init__(self,
                 raw,
                 events,
                 event_id,
                 notch_filter               = 50,
                 perform_ICA                = True,
                 highpass                   = None,
                 highpass_ICA               = 1.,
                 lowpass                    = None,
                 tmin                       = -0.5,
                 tmax                       = 2.5,
                 baseline                   = (-.5,0),
                 interpolate_bad_channels   = True,
                 ):
        super(PreprocessingPipeline,).__init__()
        np.random.seed(12345)
        
        self.raw                            = raw
        self.events                         = events
        self.event_id                       = event_id
        
        self.notch_filter                   = notch_filter
        
        self.highpass                       = highpass
        self.lowpass                        = lowpass
        self.highpass_ICA                   = highpass_ICA
        
        self.tmin                           = tmin
        self.tmax                           = tmax
        self.baseline                       = baseline
        
        self.perform_ICA                    = perform_ICA
        
        self.interpolate_bad_channels       = interpolate_bad_channels
    """
    necessary step: re-reference - explicitly
    """
    def re_refernce(self,):
        self.raw_ref ,_  = mne.set_eeg_reference(self.raw,
                                                 ref_channels     = 'average',
                                                 projection       = True,)
        self.raw_ref.apply_proj() # it might tell you it already has been re-referenced, but do it anyway
    
    """
    necessary step: notch filtering
    """
    def notch_filtering(self,):
        notch_filter = self.notch_filter
        # everytime before filtering, explicitly pick the type of channels you want
        # to perform the filters
        picks = mne.pick_types(self.raw_ref.info,
                               meg = False, # No MEG
                               eeg = True,  # YES EEG
                               eog = self.perform_ICA,  # depends on ICA
                               )
        # regardless the bandpass filtering later, we should always filter
        # for wire artifacts and their oscillations
        self.raw_ref.notch_filter(np.arange(notch_filter,241,notch_filter),
                                  picks = picks)
    
    """
    optional step: highpass, lowpass, bandpass filtering
    """
    def filtering(self,):
        lowpass = self.lowpass
        highpass = self.highpass
        
        if np.logical_and(highpass is not None,lowpass is not None):
            self.raw_ref = self.raw_ref.filter(highpass,lowpass)
        else:
            if lowpass is not None:
                self.raw_ref = self.raw_ref.filter(None,lowpass,)
            elif highpass is not None:
                self.raw_ref = self.raw_ref.filter(highpass,None)
        
        self.raw_ref_for_ICA = self.raw_ref.copy().filter(self.highpass_ICA,lowpass)
    """
    necessary step: epoching the raw, not-filtered data
    """
    def epoching(self,detrend = 1,preload = True,):
        picks = mne.pick_types(self.raw_ref.info,
                               meg = False, # No MEG
                               eeg = True,  # YES EEG
                               eog = self.perform_ICA,  # depends on ICA
                               )
        self.epochs      = mne.Epochs(self.raw_ref,
                                      self.events,    # numpy array
                                      self.event_id,  # dictionary
                                      tmin        = self.tmin,
                                      tmax        = self.tmax,
                                      baseline    = self.baseline, # range of time for computing the mean references for each channel and subtract these values from all the time points per channel
                                      picks       = picks,
                                      detrend     = detrend, # linear detrend
                                      preload     = preload # must be true if we want to do further processing
                                      )
        self.epochs_for_ICA = mne.Epochs(self.raw_ref_for_ICA,
                                         self.events,    # numpy array
                                         self.event_id,  # dictionary
                                         tmin        = self.tmin,
                                         tmax        = self.tmax,
                                         baseline    = self.baseline, # range of time for computing the mean references for each channel and subtract these values from all the time points per channel
                                         picks       = picks,
                                         detrend     = detrend, # linear detrend
                                         preload     = preload # must be true if we want to do further processing
                                         )
    """
    optional step: mark bad channels, interpolate them in necessary
    """
    
    
    def mark_bad_channels(self,
                          check_epochs = True,
                          check_epochs_for_ICA = True,
                          ):
        picks = mne.pick_types(self.raw_ref.info,
                               meg = False, # No MEG
                               eeg = True,  # YES EEG
                               eog = self.perform_ICA,  # depends on ICA
                               )
        if check_epochs:
            self.bad_channels_list_epochs,self.epochs = _check_bad_channels(self.epochs,picks)
            if self.interpolate_bad_channels:
                self.epochs.interpolate_bads()
        if check_epochs_for_ICA:
            self.bad_channels_list_epochs_for_ICA, self.epochs_for_ICA = _check_bad_channels(self.epochs_for_ICA,picks)
            if self.interpolate_bad_channels:
                self.epochs_for_ICA.interpolate_bads()
    """
    optional step: mark bad epochs
    """
    def mark_bad_epochs(self,
                        check_epochs = True,
                        check_epochs_for_ICA = True,
                        ):
        picks = mne.pick_types(self.raw_ref.info,
                               meg = False, # No MEG
                               eeg = True,  # YES EEG
                               eog = self.perform_ICA,  # depends on ICA
                               )
        if check_epochs:
            self.bad_epochs_list_epochs = faster_bad_epochs(self.epochs,picks)
        if check_epochs_for_ICA:
            self.bad_epochs_list_epochs_for_ICA = faster_bad_epochs(self.epochs_for_ICA,picks)
    """
    optional step: mark bad channels for each epoch
    """
    def mark_bad_channels_for_each_epoch(self,
                                         ):
        picks = mne.pick_types(self.raw_ref.info,
                               meg = False, # No MEG
                               eeg = True,  # YES EEG
                               eog = self.perform_ICA,  # depends on ICA
                               )
        self.bad_channels_list_epochs,self.epochs = _check_bad_channels(
                self.epochs,picks,func = faster_bad_channels_in_epochs)
        if self.interpolate_bad_channels:
            self.epochs.interpolate_bads()
        
        
    """
    half necessary half optional step: ICA fitting, ##### Not applying the ICA yet #####
    """
    def fit_ica(self,
                method                  = dict(noise_cov = 'empirical',
                                               ica       = 'fastica',),
                rank                    = None,
                n_components            = .99,
                n_pca_components        = .99,
                max_pca_components      = None,
                max_iter                = int(3e3),
                verbose                 = 1,
                ):
        picks = mne.pick_types(self.epochs_for_ICA.info,
                               meg = False, # No MEG
                               eeg = True,  # YES EEG
                               eog = False,
                               )
        # calculate the noise covariance of the epochs
        noise_cov   = mne.compute_covariance(self.epochs_for_ICA,
                                             tmin                   = self.baseline[0],
                                             tmax                   = self.baseline[1],
                                             method                 = method['noise_cov'],
                                             rank                   = rank,
                                             )
        # define an ica function
        ica         = mne.preprocessing.ICA(n_components            = n_components,
                                            n_pca_components        = n_pca_components,
                                            max_pca_components      = max_pca_components,
                                            method                  = method['ica'],
                                            max_iter                = max_iter,
                                            noise_cov               = noise_cov,
                                            random_state            = 12345,
                                            )
        # fit the ica
        ica.fit(self.epochs_for_ICA,
                picks   = picks,
                start   = self.tmin,
                stop    = self.tmax,
                decim   = 3,
                tstep   = 1., # Length of data chunks for artifact rejection in seconds. It only applies if inst is of type Raw.
                verbose = verbose, # change to False if you want to omit the print information
                )
        self.ica = ica
    """
    optional step: mark bad ica components using FASTER algorithm
    """
    def mark_bad_ica_components_by_FASTER(self,):
        self.bad_ica_list_by_FASTER = faster_bad_components(self.ica,self.epochs_for_ICA)
        for idx in self.bad_ica_list_by_FASTER:
            self.ica.exclude.append(idx)
    """
    optional step: detect bad ica components using MNE-python function
    """
    def detect_artifacts(self,
                         eog_ch         = ['TP9','TP10','PO9','PO10'],
                         eog_criterion  = 0.4, # arbitary choice
                         skew_criterion = 1,   # arbitary choice
                         kurt_criterion = 1,   # arbitary choice
                         var_criterion  = 1,   # arbitary choice
                         ):
        # search for artificial ICAs automatically
        # most of these hyperparameters were used in a unrelated published study
        self.ica.detect_artifacts(self.epochs_for_ICA,
                                  eog_ch         = eog_ch,
                                  eog_criterion  = eog_criterion,
                                  skew_criterion = skew_criterion,
                                  kurt_criterion = kurt_criterion,
                                  var_criterion  = var_criterion,
                                  )
    """
    half necessary half optional step: apply the fitted ica to the raw epochs
    this step is necessary only if ICA is fit
    """
    def apply_ica(self,):
        self.epochs = self.ica.apply(self.epochs,
                                     exclude = self.ica.exclude,
                                     )
    """
    final step: remove EOG channels from the list
    """
    def final_step(self,):
        self.clearn_epochs = self.epochs_ica.copy().pick_types(eeg = True, eog = False)








