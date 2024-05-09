import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

try:
    cv2.setNumThreads(0)
except():
    pass

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import get_contours

logging.captureWarnings(True)
logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    filename="/tmp/caiman.log",
                    level=logging.WARNING)

def extract(fname, motion_correct=True, use_cnn=False, frames_window=10, dview=None, n_processes=None):
    # dataset dependent parameters
    fr = 30                     # imaging rate in frames per second
    decay_time = 0.4            # length of a typical transient in seconds

    # motion correction parameters
    strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
    max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
    max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
    pw_rigid = True             # flag for performing non-rigid motion correction

    # parameters for source extraction and deconvolution
    p = 1                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thr = 0.85            # merging threshold, max correlation allowed
    rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6             # amount of overlap between the patches in pixels
    K = 4                       # number of components per patch
    gSig = [4, 4]               # expected half size of neurons in pixels
    method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 1                    # spatial subsampling during initialization
    tsub = 1                    # temporal subsampling during intialization

    # parameters for component evaluation
    min_SNR = 2.0               # signal to noise ratio for accepting a component
    rval_thr = 0.85             # space correlation threshold for accepting a component
    cnn_thr = 0.99              # threshold for CNN based classifier
    cnn_lowest = 0.1            # neurons with cnn probability lower than this value are rejected

    opts_dict = {'fnames': [fname],
                 'fr': fr,
                 'decay_time': decay_time,
                 'strides': strides,
                 'overlaps': overlaps,
                 'max_shifts': max_shifts,
                 'max_deviation_rigid': max_deviation_rigid,
                 'pw_rigid': pw_rigid,
                 'p': p,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub,
                 'merge_thr': merge_thr,
                 'min_SNR': min_SNR,
                 'rval_thr': rval_thr,
                 'use_cnn': use_cnn,
                 'min_cnn_thr': cnn_thr,
                 'cnn_lowest': cnn_lowest}

    opts = params.CNMFParams(params_dict=opts_dict)

    if dview is None:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                         n_processes=n_processes,
                                                         single_thread=False)

    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    return fit_file(cnm, motion_correct=motion_correct, frames_window=frames_window)

def fit_file(cnm, motion_correct=False, frames_window=10, dview=None, n_processes=None):
    fnames = cnm.params.get('data', 'fnames')
    if os.path.exists(fnames[0]):
        _, extension = os.path.splitext(fnames[0])[:2]
        extension = extension.lower()
    else:
        logging.warning("Error: File not found, with file list:\n" + fnames[0])
        raise Exception('File not found!')

    base_name = pathlib.Path(fnames[0]).stem + "_memmap_"
    if extension == '.mmap':
        fname_new = fnames[0]
        Yr, dims, T = cm.mmapping.load_memmap(fnames[0])
        if np.isfortran(Yr):
            raise Exception('The file should be in C order (see save_memmap function)')
    else:
        if motion_correct:
            print("Running motion correction")
            mc = MotionCorrect(fnames, dview=cnm.dview, **cnm.params.motion)
            mc.motion_correct(save_movie=True)
            fname_mc = mc.fname_tot_els if cnm.params.motion['pw_rigid'] else mc.fname_tot_rig
            if cnm.params.get('motion', 'pw_rigid'):
                b0 = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                        np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
                cnm.estimates.shifts = [mc.x_shifts_els, mc.y_shifts_els]
            else:
                b0 = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
                cnm.estimates.shifts = mc.shifts_rig
            b0 = 0
            print("done")
            fname_new = cm.mmapping.save_memmap(fname_mc, base_name=base_name, order='C',
                                             border_to_0=b0)
        else:
            print("Skipping motion correction")
            fname_new = cm.mmapping.save_memmap(fnames, base_name=base_name, order='C')
        Yr, dims, T = cm.mmapping.load_memmap(fname_new)

    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm.mmap_file = fname_new

    print("First fitting...")
    cnm.fit(images)
    print("done")


    print("Correlating...")
    Cn = cm.summary_images.local_correlations(images[::max(T//1000, 1)], swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    print("done")

#     cnm.save(fname_new[:-5]+'_init.hdf5')

#     print("Refitting...")
#     cnm2 = cnm.refit(images, dview=dview)

#     print("done")

    print("Evaluating components...")
    cnm.estimates.evaluate_components(images, cnm.params, dview=cnm.dview)
    print("done")

    # update object with selected components
    print("Selecting components...")
    cnm.estimates.select_components(use_object=True)
    print("done")

    print("Building contour coordinates")
    if 'csc_matrix' not in str(type(cnm.estimates.A)):
        cnm.estimates.A = scipy.sparse.csc_matrix(cnm.estimates.A)
    cnm.estimates.coordinates = get_contours(cnm.estimates.A, cnm.estimates.dims, thr=0.2, thr_method='max')
    print("done")

    # Extract DF/F values
    print("Detrending dF/F...")
    cnm.estimates.detrend_df_f(quantileMin=8, frames_window=frames_window)
    print("done")
    cnm.estimates.Cn = Cn
    cnm.save(cnm.mmap_file[:-4] + 'hdf5')

    print("Shutting down the cluster...")
    cm.cluster.stop_server(dview=cnm.dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    print("done")

    return cnm, images, dview, n_processes

def load_images(fname):
    Yr, dims, T = cm.load_memmap(fname)
    return np.reshape(Yr.T, [T] + list(dims), order='F')
