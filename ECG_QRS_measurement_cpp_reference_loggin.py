import matplotlib.pyplot as plt
import numpy as np
#from scipy import signal
#from scipy.signal import find_peaks
import scipy
#import scipy.signal


def ECG_QRS_measurement(beat_avg, medianRR, Fs = 300, log=print):

    ''' Look for:
                    - the QRS onset and offset locations
                    - the P wave peak and onset locations
        on an average beat computed on lead I or II (mostly tested on lead I).
        Input:
            beat_avg: average beat samples (mV).
            meadianRR: median RR associated to the average beat (ms).
            Fs: sampling frequency of the average beat in Hz. THIS FUNCTION ONLY SUPPORTS 300 Hz.
            log: A function to log debug info to.  Defaults to print.
        Output:
            clipped_avg_beat: windowed average beat of 'medianRR' length.
            QRS_onset: estimated QRS onset index (nb samples).
            QRS_offset: estimted QRS offset index (nb samples).
            QRS_duration: QRS duration (ms).
            CI: QRS measurement confidence index: 1 high
                                                  0 low
            p_inx: estimated P-wave location peak index (nb samples).
            p_onset: estimated P-wave onset location index (nb samples).
            pr_interval: PR interval duration (ms).
            error: error message
    Algorithm main processing steps:
    1) Select the maximum magnitude QRS point as the starting reference point for the search of the J-point and QRS onset.
    The initial reference point is assumed to be an R-wave (this decision can be subsequently overruled).
    2) Look for the J-point. The 1st step of the search consists in validating/overruling the initial peak selection. An
    alternative peak is selected if the initial reference is found to be an S-wave (or something else).
    The J-point search is done in an incremental way. The algorithm starts by checking the most common/simple beat morphologies.
    A negative search will lead to more complex checks of less common/abnormal morphologies. The search is mainly based on
    tests of the beat morphology using the 1st and 2nd derivatives.
    Both the tests conditions and the detection thresholds have been determined/tuned running tests on ~5000 average beats.
    3) Look for the QRS onset in a similar way as the J-point location (except for the beat polarity check).
    4) Look for the P wave and the P wave onset. This version of the algorithm considers only positive P waves.
    '''

    beat_avg = np.array(beat_avg)
    debug_flag = False
    # v2 --->
    # avg_beat_features = {'clipped_avg_beat': [], 'qrs_onset': [], 'qrs_offset': [], 'qrs_duration': [], 'CI': 0,
    #                        'p_inx': [], 'p_onset': [], 'pr_interval': []}
    avg_beat_features = {'clipped_avg_beat': None, 'qrs_onset': None, 'qrs_offset': None, 'qrs_duration': None, 'CI': 1,
                         'p_inx': None, 'p_onset': None, 'pr_interval': None} # <---- v2

    if Fs != 300:
        return('Unsupported sampling frequency Fs.', avg_beat_features)

    # sanity checks
    # v2 --->
    if medianRR == 0:
        return ('Abnormal HR.', avg_beat_features) # <---- v2
    HR = 1 / (medianRR * 1e-3) * 60 # bpm
    log('MeasureQRSWidth heart_rate: %s' % HR)
    if HR > 201:
        return ('Abnormal HR.', avg_beat_features)
    # elif: others to be added...

    medianRR = medianRR * 1e-3 * Fs # ms -> nb samples
    log('MeasureQRSWidth median_rr_samples: %s' % medianRR)

    try:
        beat_avg0 = beat_avg
        beat_avg = beat_avg - np.mean(beat_avg)

        avg_beat_features['clipped_avg_beat'] = beat_avg0 # already defined in case of returned error

        # b, a = signal.butter(1, [2.3 / (Fs / 2), 60 / (Fs / 2)], btype='bandpass')
        b = np.array([ 0.40841323,  0.        , -0.40841323])
        a = np.array([ 1.        , -1.14246886,  0.18317354 ])
        # remove baseline; enhance peaks (particularly in QS morphologies where the QR waves are rather a down-slope edge)
        beat_avg_bpf = scipy.signal.filtfilt(b, a, beat_avg)

        # QRS reference peak search ---------------------------------->
        # Assumption: maximum magnitude point within the search window is an R wave (this assumption can be overruled subsequently)
        inx = 180 # qrs peak index in many average beats
        wl = 25
        wrange = np.arange(inx - wl, inx + wl)

        qrs_peak_inx = np.where(np.abs(beat_avg_bpf[wrange]) == np.max(np.abs(beat_avg_bpf[wrange])))[0]
        abs_peak_max = np.abs(beat_avg_bpf[wrange][qrs_peak_inx])
        cond1 = len(qrs_peak_inx) > 1 # more than 1 peak
        cond2 = np.max(beat_avg0[wrange]) < -0.2 # peak at unexpected location
        # other conditions...

        if cond1 or cond2:
            return ('QRS measurement error.', avg_beat_features)
        else:
            qrs_peak_inx = qrs_peak_inx[0]

        qrs_peak_p_inx = np.where(beat_avg_bpf[wrange] == np.max(beat_avg_bpf[wrange]))[0][0] # positive peak
        qrs_peak_n_inx = np.where(beat_avg_bpf[wrange] == np.min(beat_avg_bpf[wrange]))[0][0] # negative peak

        qrs_peak_inx = int(inx - wl + qrs_peak_inx)
        qrs_peak_p_inx = int(inx - wl + qrs_peak_p_inx)
        qrs_peak_n_inx = int(inx - wl + qrs_peak_n_inx)

        p1 = np.min([qrs_peak_p_inx, qrs_peak_n_inx]) # leftmost peak
        p2 = np.max([qrs_peak_p_inx, qrs_peak_n_inx]) # rightmost peak

        log('MeasureQRSWidth peak search:')
        log('\t:abs_peak_ix = %s' % qrs_peak_inx)
        log('\t:peak_l = %s' % p1)
        log('\t:peak_r = %s' % p2)
        inv_polarity = False
        if beat_avg[qrs_peak_inx] < 0:
            beat_avg = -beat_avg
            inv_polarity = True
        log('MeasureQRSWidth inv_polarity: %s' % inv_polarity)

        # <----------------------- QRS reference peak search

        # beat amplitude normalization used to reduce the dispersion of the derivatives magnitude range
        norm_coeff = np.abs(beat_avg[qrs_peak_inx])
        norm_coeff2 = norm_coeff
        log('MeasureQRSWidth norm_coeff: %s' % norm_coeff)
        beat_wave_lpf_norm = beat_avg / norm_coeff
        beat_wave_diff = np.diff(beat_wave_lpf_norm)

        # beat clipping using medianRR
        inxs = int(np.max([qrs_peak_inx - 0.42 * medianRR, 0]))
        inxe = int(np.min([len(beat_wave_lpf_norm), inxs + medianRR]))
        qrs_peak_inx = qrs_peak_inx - inxs
        beat_wave_lpf_norm = beat_wave_lpf_norm[inxs:inxe]
        beat_wave_diff = beat_wave_diff[inxs:inxe]
        beat_avg0 = beat_avg0[inxs:inxe]


        # quality assessment of the average beat ------------------->
        # 'distor_ratio': baseline variation w.r.t. the QRS estimated amplitude
        # 'noise_var': average beat 'smoothness' estimation
        distor_ratio = (np.abs(beat_avg0[0]) + np.abs(beat_avg0[-1])) / (beat_avg0[qrs_peak_p_inx - inxs] - beat_avg0[qrs_peak_n_inx - inxs])
        log('MeasureQRSWidth quality assessment distort_ratio: %s' % distor_ratio)

        if debug_flag:
            print('distor ratio : {:.2f}'.format(distor_ratio*100))

        try:
            noise_var = len(np.where(np.diff(np.sign(np.diff(beat_avg0))) != 0)[0])
            noise_ratio = noise_var / len(beat_avg0)
            if debug_flag:
                print('noise ratio: {:.2f}'.format(noise_ratio*100))
        except:
            noise_ratio = 0
            if debug_flag:
                print('noise ratio: {:.2f}'.format(noise_ratio*100))
        log('MeasureQRSWidth quality assessment noise_ratio: %s' % noise_ratio)

        # v2 --->
        # look for potential distorted/noisy beats, or beats where we are not confident on an accurate measurement (unusual morphologies)
        low_ci_flag_1 = False; low_ci_flag_2 = False
        if (distor_ratio * 100 >= 4.7 and noise_ratio * 100 >= 11.8):
            log('MeasureQRSWidth quality assessment.1')
            low_ci_flag_1 = True # discard only if p-wave not found
        if (distor_ratio * 100 >= 14 and noise_ratio * 100 >= 12):
            log('MeasureQRSWidth quality assessment.2')
            low_ci_flag_2 = True # discard whether or not a p-wave is found
        if (distor_ratio * 100 >= 37 and noise_ratio * 100 >= 10) or (distor_ratio * 100 >= 9 and noise_ratio * 100 >= 16):
            log('MeasureQRSWidth quality assessment.3')
            low_ci_flag_2 = True  # discard whether or not a p-wave is found

        # if (distor_ratio * 100 >= 5.5 and noise_ratio * 100 >= 6.5) or \
        if (distor_ratio * 100 >= 6.5 and noise_ratio * 100 >= 6.9) or \
                (beat_avg0[qrs_peak_p_inx - inxs] - beat_avg0[qrs_peak_n_inx - inxs]) < 0.25 and distor_ratio * 100 > 19 or \
                (beat_avg0[qrs_peak_p_inx - inxs] - beat_avg0[qrs_peak_n_inx - inxs]) < 0.33 and distor_ratio * 100 > 35:
            # avg_beat_features['CI'] = 0 # potential low quality beat; QRS and PR measurements should be discarded
            log('MeasureQRSWidth quality assessment.4')
            low_ci_flag_1 = True # discard only if p-wave not found
        # else:
        #     avg_beat_features['CI'] = 1 # good quality # <--- v2


        avg_beat_features['distor_ratio'] = distor_ratio
        avg_beat_features['noise_ratio'] = noise_ratio
        # < ----------------------- quality assessment of the average beat




        # J-point location ------------------------------------------------------------------------------------>

        QS_flag = False
        thresh = 0.0113333

        aux = beat_wave_diff[qrs_peak_inx + 1:]

        # v2 ---->
        aux3 = beat_wave_lpf_norm[qrs_peak_inx + 1:] # <----- v2

        # If the reference peak doesn't look like an R-wave, switch to the 2nd peak option.
        inx_aux = np.where(aux >= 0)[0][0]
        try:
            # ndev_test = len(np.where(aux[:65] < 0)[0]) / len(aux[:65]) > 0.70
            ndev_test = len(np.where(aux[:75] < 0)[0]) / len(aux[:75]) > 0.62
        except:
            ndev_test = False

        slope_test2 = (beat_wave_lpf_norm[p2 - inxs + 1] - beat_wave_lpf_norm[p2 - inxs]) * (beat_wave_lpf_norm[p2 - inxs] - beat_wave_lpf_norm[p2 - inxs - 1])

        aux2 = np.diff(aux) # 2nd derivative
        cond1 = (inx_aux > 38 or (inx_aux > 28 and HR > 97) or ndev_test ) and np.min(aux[20:57]) < -0.00148 and slope_test2 < 9.2e-4
        cond2 = np.min(aux[30:130]) <= -1.84e-3
        cond3 = not (np.min(aux2[1:10]) < -1.5e-2 and np.max(aux[1:10]) < 0 and len(np.where(np.diff(np.sign(aux2[0:10])) < 0)[0]) > 1 and inx_aux < 55)
        # cond4 = not (qrs_peak_inx == p1 - inxs)
        cond4 = qrs_peak_inx == p2 - inxs

        # v2 ---->
        cond5 = not (np.mean(aux3[35:70]) < -0.23 and np.mean(abs(aux[35:70])) < 0.005) # <----- v2

        # if  cond1 and cond2 and cond3:
        #     print('test')


        # ----------------------------------------------------------------
        # v2 -->
        # if cond1 and cond2 and cond3 and cond4: # the main reasons for all these conditions to be true (but not exclusively)
                                      # are:
                                      #     - an average beat with inverted polarity
                                      #     - an initial peak selection of an S-wave
                                      #     - QRS with QS morphology.
        if cond1 and cond2 and cond3 and cond4 and cond5: # <---- v2
            log('MeasureQRSWidth J-Point.1')
            # the main reasons for all these conditions to be true (but not exclusively)
            # are:
            #     - an average beat with inverted polarity
            #     - an initial peak selection of an S-wave
            #     - QRS with QS morphology.


            # if qrs_peak_inx == p2 - inxs: # v4
            #     log('MeasureQRSWidth J-Point.1.1')
            qrs_peak_inx = p1 - inxs
            norm_coeff2 = np.abs(beat_avg[qrs_peak_inx + inxs])
            QS_flag = True  # QS-like morphology
            # else: # v4
            #     log('MeasureQRSWidth J-Point.1.2')
            #     qrs_peak_inx = p2 - inxs
            #     norm_coeff2 = np.abs(beat_avg[qrs_peak_inx + inxs])


            beat_wave_diff = -beat_wave_diff  # inverse polarity
            aux = beat_wave_diff[qrs_peak_inx + 1:]
            beat_wave_lpf_norm = -beat_wave_lpf_norm
            inv_polarity = not inv_polarity
        # ---------------------------------------------------------------


        if debug_flag:
            plt.figure()
            plt.plot(beat_wave_lpf_norm, '.-')
            plt.plot(qrs_peak_inx, beat_wave_lpf_norm[qrs_peak_inx], 'o')


        # if np.min(aux[3:10]) > 1e6: # v3: remove J-point 2 v4 (J-Point.3 renamed J-Point.2)
        # # if np.min(aux[3:10]) > -5e-3:  # QS morphology check; if true, J-point ~= peak index. Search finished.
        #     log('MeasureQRSWidth J-Point.2')
            inxTot = 0
        # else: # Most general case. Start search from R-wave peak. # v4
        log('MeasureQRSWidth J-Point.2') # J-Point.3 renamed J-Point.2
        inx0 = np.where(aux[:56] == np.min(aux[:56]))[0][0] # start search at minimum slope point

        inxTot = inx0 + 1
        aux = aux[inx0:]
        high_slope_flag = False

        # 1st pass --------------------------------
        # in most cases, it should find the S-wave, that could also be the J-point.
        inx1 = np.where(aux > 0)[0]  # derivative changes sign
        # if len(inx1) == 0:
        #     log('MeasureQRSWidth J-Point.2.1') # remove J-Point.2.1: v4
        #     return ('QRS measurement error.', avg_beat_features)

        inx2 = np.where(np.abs(aux) <= thresh)[0] # derivative close to 0
        if len(inx2) == 0:  # quickly oscillating baseline
            inx2 = [np.Inf]


        inx = int(np.min([inx1[0], inx2[0]]))
        inxTot += inx
        aux = aux[inx:]
        # -----------------------------------------

        # slurring QRS
        slur_flag = False
        if (np.min(aux[1:6]) < -0.0086 and np.max(aux[:5]) < 0.058 and inx_aux <= 50) or np.min(aux[:6]) < -0.015:
            log('MeasureQRSWidth J-Point.2.1') # rename J-Point.2.2 -> J-Point.2.1: v4
            aux_tmp = aux
            inxTotCpy = inxTot

            inxTot += 5
            aux = aux[5:]


            inx1 = np.where(aux > 0)[0]
            if len(inx1) == 0:
                inx1 = [np.Inf]

            inx2 = np.where(np.abs(aux) < 4.5e-3)[0]
            if len(inx2) == 0:
                inx2 = [np.Inf]

            inx = int(np.min([inx1[0], inx2[0]]))
            if inx < 14 and inx1[0] < 51:
                log('MeasureQRSWidth J-Point.2.1.1')
                inxTot += inx
                aux = aux[inx:]

            slur_flag = True


            if inx == 0 and inx2[0] == inx: # rollback
                log('MeasureQRSWidth J-Point.2.1.2')
                aux = aux_tmp
                inxTot = inxTotCpy
                slur_flag = False

        # After S-wave
        if norm_coeff > 0.79:
            thresh1 = 0.00634
            log('MeasureQRSWidth J-Point.2.2 thresh1: %s' % thresh1)
        else:
            thresh1 = 0.0061
            log('MeasureQRSWidth J-Point.2.3 thresh1: %s' % thresh1)
        if np.max(aux[:9]) > thresh1 and not slur_flag:  # additional non-negligible slope increase
            log('MeasureQRSWidth J-Point.2.4')
            aux_tmp = aux
            inxTotCpy = inxTot

            inx = np.where(np.max(aux[:9]) == aux[:9])[0][0]
            inxTot += inx + 1
            aux = aux[inx + 1:]

            inx1 = np.where(aux < 0)[0]  # derivative changes sign
            if len(inx1) == 0:
                inx1 = [np.Inf]

            if norm_coeff > 0.21:
                thresh1 = 0.031
                log('MeasureQRSWidth J-Point.2.4.1 thresh1: %s' % thresh1)
            else:
                thresh1 = 0.0184
                log('MeasureQRSWidth J-Point.2.4.2 thresh1: %s' % thresh1)

            inx2 = np.where(np.abs(aux) <= thresh1)[0] # derivative close to 0
            if len(inx2) == 0:
                inx2 = [np.Inf]

            inx = int(np.min([inx1[0], inx2[0]]))

            if inx < 2:
                log('MeasureQRSWidth J-Point.2.4.3')
                if norm_coeff > 0.74:
                    thresh1 = 1e-2
                    log('MeasureQRSWidth J-Point.2.4.3.1 thresh1: %s' % thresh1)
                elif norm_coeff > 0.57:
                    thresh1 = 7e-3
                    log('MeasureQRSWidth J-Point.2.4.3.2 thresh1: %s' % thresh1)
                else:
                    thresh1 = 5e-3
                    log('MeasureQRSWidth J-Point.2.4.3.3 thresh1: %s' % thresh1)
                inx = np.where(np.abs(aux) <= thresh1)[0]
                if len(inx) == 0:
                    inx = 0
                else:
                    inx = inx[0]

            if inx > 12:  # restart search with a higher detection threshold; overrule previous inx
                log('MeasureQRSWidth J-Point.2.4.4')
                high_slope_flag = True
                inx = np.where(np.abs(aux) <= 2.1e-2)[0]
                if len(inx) == 0:
                    inx = np.Inf
                else:
                    inx = inx[0]

                if inx > 10:  # restart search with a higher detection threshold; overrule previous inx
                    log('MeasureQRSWidth J-Point.2.4.4.1')
                    high_slope_flag = True
                    inx = np.where(np.abs(aux) <= 1e-1)[0]
                    if len(inx) == 0:
                        inx = np.Inf
                    else:
                        inx = inx[0]

                    # if inx > 11:  # restart search with a higher detection threshold; overrule previous inx: v4
                    #     log('MeasureQRSWidth J-Point.2.4.4.1.1')
                    #     high_slope_flag = True
                    #     inx = np.where(np.abs(aux) <= 5e-1)[0]
                    #     if len(inx) == 0:
                    #         log('MeasureQRSWidth J-Point.2.4.4.1.1.1')
                    #         inx = np.Inf
                    #     else:
                    #         inx = inx[0]

            inxTot += inx
            aux = aux[inx:]

            if (inx == 0 and slur_flag == False) or (inx == 0 or inx == 1 and high_slope_flag == True):  # rollback
                log('MeasureQRSWidth J-Point.2.4.5')
                high_slope_flag = False
                aux = aux_tmp
                inxTot = inxTotCpy


        # multiple types of wide QRS morphologies...

        # wide QRS (1)
        if np.min(aux[6:15]) < -4.5e-2:
            log('MeasureQRSWidth J-Point.2.5')
            inxTotCpy = inxTot
            inx = np.where(aux[6:15] == np.min(aux[6:15]))[0][0]
            inxTot += inx + 6
            aux = aux[inx + 6:]
            inx1 = np.where(np.abs(aux) <= thresh)[0]
            if len(inx1) == 0:
                inx1 = [np.Inf]
            inx2 = np.where(aux > 0)[0]
            if len(inx2) == 0:
                inx2 = [np.Inf]
            inx = int(np.min([inx1[0], inx2[0]]))
            if inx < 60:
                log('MeasureQRSWidth J-Point.2.5.1')
                inxTot += inx
                aux = aux[inx:]
            # else:  # rollback # v4
            #     log('MeasureQRSWidth J-Point.2.5.2')
            #     inxTot = inxTotCpy

        # double_bump_flag = False # v4

        if norm_coeff >= 0.19:
            # thresh2 = -0.025 # v3
            thresh2 = -0.0245 # v3
            log('MeasureQRSWidth J-Point.2.6 thresh2: %s' % thresh2)
        else:
            thresh2 = -0.0105
            log('MeasureQRSWidth J-Point.2.7 thresh2: %s' % thresh2)

        # wide QRS (2)
        log('MeasureQRSWidth J-Point.2 np.max(aux[8:21]): %s' % np.max(aux[8:21]))
        log('MeasureQRSWidth J-Point.2 np.min(aux[20:50]): %s' % np.min(aux[20:50]))
        log('MeasureQRSWidth J-Point.2 high_slope_flag: %s' % high_slope_flag)
        log('MeasureQRSWidth J-Point.2 np.max(aux[:8]): %s' % np.max(aux[:8]))

        if (HR < 183 and np.max(aux[8:21]) > 1.965e-2 and np.min(aux[20:50]) > thresh2 and not high_slope_flag) or \
                (np.max(aux[:8] > 9.3e-2) and HR < 134):
            log('MeasureQRSWidth J-Point.2.8')

            aux_tmp = aux
            inxTotCpy = inxTot
            inx = np.where(aux[8:20] == np.max(aux[8:20]))[0][0]
            inxTot += inx + 8
            aux = aux[inx + 8:]


            # v2 ------>
            # if QS_flag and aux[0] < 0.1:
            if QS_flag and aux[0] < 7.6e-2: # <----- v2
                thresh2 = 5e-3
                log('MeasureQRSWidth J-Point.2.8.1 thresh2: %s' % thresh2)
            elif norm_coeff >= 0.38 or np.max(aux_tmp[:8]) > 8e-2:
                thresh2 = 0.02
                log('MeasureQRSWidth J-Point.2.8.2 thresh2: %s' % thresh2)
            else:
                thresh2 = 0.00375
                log('MeasureQRSWidth J-Point.2.8.3 thresh2: %s' % thresh2)

            inx1 = np.where(np.abs(aux) <= thresh2)[0]
            # if len(inx1) == 0: # v4
            #     inx1 = [np.Inf]
            # elif inx1[0] == 0: # v4
            #     log('MeasureQRSWidth J-Point.2.8.4')
            #     inx1 = np.where(np.abs(aux) <= 5e-3)[0]
            #     if len(inx1) == 0:
            #         inx1 = [np.Inf]

            inx2 = np.where(aux < 0)[0]
            if len(inx2) == 0:
                inx2 = [np.Inf]
            inx = int(np.min([inx1[0], inx2[0]]))
            inxTot += inx

            aux = aux[inx:]


            if inx >= 23 or (inx >= 19 and QS_flag):  # rollback
                log('MeasureQRSWidth J-Point.2.8.4')
                aux = aux_tmp
                inxTot = inxTotCpy

        # wide QRS (3)
        elif np.min(aux[:10]) < -3e-2 and np.max(aux[10:20]) < 0.075 and inx_aux < 64 and np.where(aux[1:] > 0)[0][0] < 36:
            log('MeasureQRSWidth J-Point.2.9')
            inx = np.where(aux[:10] == np.min(aux[:10]))[0][0]
            inxTot += inx
            aux = aux[inx:]

            inx1 = np.where(aux > 0)[0]  # derivative changes sign
            if len(inx1) == 0:
                inx1 = [np.Inf]

            inx2 = np.where(np.abs(aux) <= 1.1 * thresh)[0]
            if len(inx2) == 0:
                inx2 = [np.Inf]

            inx = int(np.min([inx1[0], inx2[0]]))
            inxTot += inx

            # double_bump_flag = True # v4

        # wide QRS (4)
        # v2 ----->
        # if np.max(aux[:20]) > 0.555:
        if np.max(aux[:25]) > 0.155: # <----- v2
            inx1 = np.where(np.max(aux[:20]) == aux[:20])[0][0]
            log('MeasureQRSWidth J-Point.2.10 inx1: %s' % inx1)
            aux_tmp = aux[inx1: inx1 + 20]
            inx = np.where(np.abs(aux_tmp) <= 4e-2)[0]
            if len(inx) > 0:
                log('MeasureQRSWidth J-Point.2.10.1 inx: %s' % inx[0])
                inx = inx[0] + inx1
                inxTot += inx

            # # 2nd slurring: v4
            # if np.min(aux[:5]) < -0.01450 and not double_bump_flag:
            #     log('MeasureQRSWidth J-Point.2.10.2')
            #     inx1 = np.where(aux > 0)[0]
            #     if len(inx1) == 0:
            #         inx1 = [np.Inf]
            #     else:
            #         inx1 = inx1[inx1 > 0]
            #
            #     inx2 = np.where(np.abs(aux) < 4.5e-3)[0]
            #     if len(inx2) == 0:
            #         inx2 = [np.Inf]
            #     else:
            #         inx2 = inx2[inx2 > 2]
            #         inx = int(np.min([inx1[0], inx2[0]]))
            #         if inx < 19:
            #             log('MeasureQRSWidth J-Point.2.10.2.1')
            #             inxTot += inx

        jp = qrs_peak_inx + 1 + inxTot
        log('MeasureQRSWidth J-Point jp: %s' % jp)

        # print('jp: {}'.format(jp))
        # <------------------------------------------ J-point location

        if debug_flag:
            plt.plot(jp, beat_wave_lpf_norm[jp], 'o')

        # ---------------------------------------- QRS onset ----------------------------------------
        norm_coeff = norm_coeff2
        aux = beat_wave_diff[:qrs_peak_inx - 1]
        aux = np.array(aux)
        aux = -aux[::-1]
        log('MeasureQRSWidth QRS onset  norm_coeff: %s' % norm_coeff)

        if np.min(aux[:20]) <= -1.1e-2 and QS_flag and beat_wave_lpf_norm[qrs_peak_inx] > 0.08: # overrule QS-like morphology detection
            log('MeasureQRSWidth QRS onset.1')
            QS_flag = False

        # if QS_flag and np.min(aux[:3]) > -0.04:
        if QS_flag and np.min(aux[:3]) > -0.038: # QS-like morphology; use current index as QRS onset; search finished.
            inxTot = 0
            log('MeasureQRSWidth QRS onset.2')
        else: # at this point, in most cases the search should be starting from an R-wave peak
            log('MeasureQRSWidth QRS onset.3')
            inx0 = np.where(aux[:46] == np.min(aux[:46]))[0][0]
            inx0_1 = np.where(aux >= 0)[0]
            if len(inx0_1) > 0:
                log('MeasureQRSWidth QRS onset.3.1')
                inx0_1 = inx0_1[0] - 1
                if inx0_1 < 1:
                    log('MeasureQRSWidth QRS onset.3.1.1')
                    inx0_1 = np.Inf
            else:
                inx0_1 = np.Inf

            inx0 = int(np.min([inx0_1, inx0]))

            # if inx0 > 43 or (beat_wave_lpf_norm[qrs_peak_inx] < 0 and inx0 == inx0_1): # QS type; v4
            #     log('MeasureQRSWidth QRS onset.3.2')
            #     inx0 = 0

            inxTot = inx0 + 1
            aux = aux[inx0:]

            # 1st pass
            aux2 = np.diff(aux) # 2nd derivative
            inflexion = False
            inx1 = np.where(aux > 0)[0]  # derivative changes sign
            if len(inx1) == 0:
                inx1 = [np.Inf]

            # v2 --->
            # if norm_coeff > 0.46:
            if norm_coeff > 0.65:
                thresh1 = 0.038 # <--- v2
                log('MeasureQRSWidth QRS onset.3.2 thresh1: %s' % thresh1)
            else:
                thresh1 = 0.01
                log('MeasureQRSWidth QRS onset.3.3 thresh1: %s' % thresh1)

            inx2 = np.where(np.abs(aux) <= thresh1)[0]
            if len(inx2) == 0:
                inx2 = [np.Inf]


            inx3 = np.where(aux2 < 5e-4)[0]
            if len(inx3) < 3:
                inx3 = [np.Inf]
            else:
                inx3 = inx3[inx3 > 8]

            inx = int(np.min([inx1[0], inx2[0], inx3[0]]))
            inxTot += inx
            aux = aux[inx:] # QRS complex with no Q deflection

            # if inx == inx3[0]:
            if inx == inx3[0] and np.max(aux[:15]) < 0.118:
                log('MeasureQRSWidth QRS onset.3.4')
                inflexion = True

            thresh1 = 0.0049

            # 2nd pass
            # if np.max(aux[:12]) >= thresh1 and not inflexion:
            if np.max(aux[:12]) >= thresh1: # look for Q deflection
                log('MeasureQRSWidth QRS onset.3.5')
                # v2 --->
                aux_tmp = aux # <--- v2
                inx1 = np.where(aux[:12] == np.max(aux[:12]))[0][0]
                aux = aux[inx1:]
                aux2 = np.diff(aux)

                # print(aux[:10])
                # print(aux2[:10])

                # v3 ->
                # thresh1 = 2.6e-3
                thresh1 = 2.8e-3
                # <- v3


                inx2 = np.where(np.abs(aux) <= thresh1)[0]
                if len(inx2) == 0:
                    inx2 = [np.Inf]

                inx3 = np.where(aux < 0)[0]
                if len(inx3) == 0:
                    inx3 = [np.Inf]


                # # inx4 = np.where(aux2 >= 0)[0]
                # inx4 = np.where(aux2 >= 1.5e-3)[0]
                # inx4 = np.where(aux2 >= 7e-4)[0] # v3
                inx4 = np.where(aux2 >= 3e-4)[0] # v3
                # print(inx4)
                if len(inx4) > 0 and np.min(inx4) >= 3:
                    log('MeasureQRSWidth QRS onset.3.6')
                    # if len(inx4) >= 4:
                    inx4 = inx4[inx4 >= 3][0] - 3
                    if aux[inx4] < 2.5e-3:
                        log('MeasureQRSWidth QRS onset.3.6.1')
                        inx4 = np.Inf
                else:
                    log('MeasureQRSWidth QRS onset.3.7')
                    inx4 = np.Inf

                inx5 = int(np.min([inx2[0], inx3[0], inx4]))

                # if inx5 == inx4:
                #     print('test ')

                # print((inx1,inx2,inx3,inx4,inx5))

                inx = inx1 + inx5

                # if (beat_wave_lpf_norm[qrs_peak_inx - 1 - inxTot] < -0.74):
                #     print('')
                # inx < 17 (21)
                # np.min(aux[inx5:inx5 + 10]) > -6.4e-3
                # v2 ---->
                # if (inx < 17 and np.max(aux[inx5 + 1:inx5 + 1 + 5]) < 0.0125 and np.min(aux[inx5:inx5 + 10]) > -6.4e-3 and inx5 > 1)  or \
                # if (inx <= 19 and np.max(aux[inx5 + 1:inx5 + 1 + 5]) < 0.0125 and np.min(aux[inx5:inx5 + 18]) > -6.3e-3 and inx5 > 1) or \ # v3
                if (inx <= 19 and np.max(aux[inx5 + 1:inx5 + 1 + 5]) < 0.0125 and np.min(aux[inx5:inx5 + 18]) > -5.5e-3 and inx5 > 1) or \
                        (beat_wave_lpf_norm[qrs_peak_inx] > 1.5 and np.min(aux[inx5:inx5 + 17]) > -0.013) or \
                        (beat_wave_lpf_norm[qrs_peak_inx - 1 - inxTot] < -0.74):  # <--- v2
                    log('MeasureQRSWidth QRS onset.3.8')
                    inxTot += inx - 2

                # v2 ----->
                # bug correction
                else: # rollback
                    log('MeasureQRSWidth QRS onset.3.9')
                    aux = aux_tmp # <---- v2

                    # 3rd condition looks for a QS morphology when the starting search point is to the right of the S wave

            # v2 ---->
            # elif np.min(aux[1:12]) < -0.0176 and np.max(aux[2:15]) < 6e-4 and norm_coeff > 0.40:  # slurry QRS
            elif np.min(aux[1:12]) < -0.0176 and np.max(aux[2:15]) < -3e-3 and norm_coeff > 0.40:  # slurry QRS # <---- v2
                log('MeasureQRSWidth QRS onset.4')
                aux_tmp = aux
                inxTotCpy = inxTot

                inx = np.where(np.abs(aux) <= 6e-3)[0]
                inx = inx[inx > 5]
                inxTot += inx[0]
                # 2nd pass
                aux = aux[inx[0]:]
                # if np.min(aux[:20]) < -3.5e-2: # v4
                #     log('MeasureQRSWidth QRS onset.4.1')
                #     # print ("IN SUBCODE -3.5e-2")
                #     inx0 = np.where(np.min(aux[:20]) == aux[:20])[0][0]
                #     aux = aux[inx0:]
                #     inx = np.where(np.abs(aux) <= 1e-2)[0]
                #     inxTot += inx0 + inx[0]

                inx = np.where(aux >= 0)[0]

                if len(inx) == 0:
                    inx = [np.Inf]
                # v2 ---->
                # if inx[0] > 15:
                if inx[0] > 15 and np.min(aux[:20]) > -9.3e-3: # <---- v2
                    log('MeasureQRSWidth QRS onset.4.1')
                    aux = aux_tmp
                    inxTot = inxTotCpy

            # multiwave or QS-like morphologies
            if (aux[0] > 0.17 and np.min(aux[:15]) < -0.11) or \
                    (aux[0] > 0.17 and np.min(aux[:15]) < -0.05 and qrs_peak_inx == (p2 - inxs)) or \
                    (aux[0] > 0.05 and np.min(aux[:15]) < -0.079) or (aux[0] > 0.06 and np.min(aux[:15]) < -0.035):
                log('MeasureQRSWidth QRS onset.5')
                inx0 = np.where(aux[:15] == np.min(aux[:15]))[0][0]
                inx1 = np.where(np.abs(aux[inx0:]) < 1e-2)[0][0]

                if inx0 + inx1 < 36:
                    log('MeasureQRSWidth QRS onset.5.1')
                    inxTot += inx0 + inx1
                    aux = aux[inx0 + inx1:]
                    if np.max(aux[:6] > 0.098):
                        log('MeasureQRSWidth QRS onset.5.2')
                        inx0 = np.where(aux[:6] == np.max(aux[:6]))[0][0]
                        inx1 = np.where(np.abs(aux[inx0:]) < 1e-2)[0][0]
                        inxTot += inx0 + inx1

        # print('inxtot: {}'.format(inxTot))
        qrs_start_inx = qrs_peak_inx - 1 - inxTot
        log('MeasureQRSWidth QRS onset qrs_start_inx: %s' % qrs_start_inx)

        # v2 - -->
        # if np.abs(beat_avg0[qrs_start_inx]) > 0.169: # potential baseline distortion
        #     avg_beat_features['CI'] = 0 # <---- v2

        if debug_flag:
            plt.plot(qrs_start_inx, beat_wave_lpf_norm[qrs_start_inx], 'o')

        QRS_int = (jp - qrs_start_inx) / Fs * 1e3  # ms

        log('MeasureQRSWidth QRS onset qrs_interval: %s' % QRS_int)

        # <----------------------------------------------------- QRS onset

        if debug_flag:
            print('QRS_int: {:}'.format(QRS_int))
            plt.figure(); plt.grid(True)
            plt.plot(beat_avg0, '.-')
            plt.plot(qrs_peak_inx, beat_avg0[qrs_peak_inx], 'o')
            plt.plot(jp, beat_avg0[jp], 'o')
            plt.plot(qrs_start_inx, beat_avg0[qrs_start_inx], 'o')

        # P-wave location --------------------------------------------------------------------------------------->
        negative_p_det = False
        if abs_peak_max > 2.62:
            # # pr_threshold = 0.15
            # pr_threshold = 0.11
            pr_threshold = 0.089
            log('MeasureQRSWidth P Wave.1 pr_threshold: %s' % pr_threshold)
        elif abs_peak_max > 0.95:
            pr_threshold = 0.029
            log('MeasureQRSWidth P Wave.2 pr_threshold: %s' % pr_threshold)
        elif abs_peak_max > 0.64:
            pr_threshold = 0.017
            log('MeasureQRSWidth P Wave.3 pr_threshold: %s' % pr_threshold)
        elif abs_peak_max < 0.127:
            # pr_threshold = 0.0126
            pr_threshold = 0.0069
            log('MeasureQRSWidth P Wave.4 pr_threshold: %s' % pr_threshold)
        else:
            # pr_threshold = 0.011
            pr_threshold = 0.008
            log('MeasureQRSWidth P Wave.5 pr_threshold: %s' % pr_threshold)

        polarity = -np.sign(int(inv_polarity) - 0.9) # False -> 1; True -> -1
        search_shift = 7
        aux = polarity * beat_avg0[0:qrs_start_inx - search_shift]
        aux = aux[::-1]
        peaks, peaks_properties = scipy.signal.find_peaks(aux, prominence=pr_threshold, height=-0.5, distance=10)
        peaks2, peaks_properties2 = scipy.signal.find_peaks(-aux, prominence=pr_threshold, height=-0.5, distance=10)

        # print(peaks)
        # print(peaks_properties)


        if debug_flag:
            plt.figure()
            plt.plot(polarity * beat_avg0, '.-')
            plt.plot(aux, '.-')



        # the code below is commented out because for now I'm looking only for positive P waves.
        # We could see negative P-waves:
        #   - for physiological reasons (very rare)
        #   - if the processed average beat polarity is inverted.
        # The last scenario mostly happens in some cases of T-wave inversion or QS-type morphologies.

        # if len(peaks) == 1:
        #     width = signal.peak_widths(aux, peaks, rel_height=0.5)[0][0]
        # if len(peaks2) == 1:
        #     width2 = signal.peak_widths(-aux, peaks2, rel_height=0.5)[0][0]

        # cond1 = len(peaks) == 0
        # cond2 = len(peaks) == 1 and peaks[0] < 5
        # cond3 = len(peaks2) == 1 and len(peaks) == 1 and peaks_properties2['prominences'][0]/peaks_properties['prominences'][0] > 3
        # cond4 = len(peaks2) == 1 and len(peaks) == 1 and width2 < width and peaks_properties2['peak_heights'][0] > peaks_properties['peak_heights'][0] \
        #         and peaks_properties2['prominences'] >= peaks_properties['prominences'] and peaks2[0] > 12
        # cond5 = len(peaks2) == 1 and len(peaks) == 1 and peaks_properties['prominences'][0]/peaks_properties2['prominences'][0] > 3
        #
        #
        # if (cond1 or cond2 or cond3 or cond4) and negative_p_det:
        #     polarity = -polarity
        #     aux = -aux
        #     peaks = peaks2
        #     peaks_properties = peaks_properties2


        cond_det = (len(peaks) + len(peaks2)) > 0
        # cond_det_2 = cond_det and np.max(list(np.concatenate([peaks_properties['prominences'], peaks_properties2['prominences']]))) > 0.12 # when negative P-waves are considered
        # cond_det_3 = cond_det and np.max(list(np.concatenate([peaks_properties['prominences'], peaks_properties2['prominences']]))) / pr_threshold > 4 # when negative P-waves are considered
        cond_det_4 = not (len(peaks) >= 3)
        cond_det_5 = not (len(peaks) == 0 and negative_p_det == False)


        # discard measurements with apparent low-quality average beats
        if cond_det_4 and cond_det and cond_det_5:
            log('MeasureQRSWidth P Wave.6')

            inx0 = np.where(peaks_properties['peak_heights'] == np.max(peaks_properties['peak_heights']))[0][0]
            p_inx = peaks[inx0]
            inxTot = p_inx

            cond2 = peaks_properties['prominences'][inx0] < 0.018 and polarity * beat_avg0[qrs_start_inx] > 1.12 * peaks_properties['peak_heights'][inx0]
            cond4 = (len(aux) - p_inx) <= 11
            if len(peaks) == 2:
                log('MeasureQRSWidth P Wave.6.1')
                cond6 = (np.max(peaks_properties['prominences']) / np.min(peaks_properties['prominences']) < 1.5 )
            else:
                log('MeasureQRSWidth P Wave.6.2')
                cond6 = False

            # discard cases with suspected false detections (1)
            if not (cond2 or cond4 or cond6):
                log('MeasureQRSWidth P Wave.6.3')

                aux2 = np.diff(aux)

                aux2 = aux2[p_inx:]
                inx = np.where(aux2[:25] == np.min(aux2[:25]))[0][0]
                inxTot += inx
                aux2 = aux2[inx:]
                inx = np.where(aux2 >= 1.2 * 1/3 * aux2[0])[0]

                if len(inx) == 0 or inx[0] > 24:
                    log('MeasureQRSWidth P Wave.6.3.1')
                    inx = np.where(aux2 >= 0.6 * aux2[0])[0]

                if len(inx) == 0:
                    log('MeasureQRSWidth P Wave.6.3.2')
                    inx = [len(aux2)]

                if len(inx) > 0:
                    inxTot += inx[0]
                    p_start_inx = qrs_start_inx - inxTot
                    p_inx = qrs_start_inx - p_inx

                    # slurry baseline
                    bl_level = np.min(polarity * beat_avg0[p_inx:qrs_start_inx])
                    if polarity * beat_avg0[p_start_inx] < bl_level:
                        log('MeasureQRSWidth P Wave.6.3.3')
                        inx = np.where(polarity * beat_avg0[p_start_inx:] >= bl_level)[0][0]
                        p_start_inx += inx - 1


                    cond2_2 = (p_inx - p_start_inx) <= 7
                    cond2_3 = np.abs(beat_avg0[p_start_inx - search_shift - 1] - beat_avg0[p_inx - search_shift - 1]) < 0.009
                    cond2_5 = np.abs(beat_avg0[p_start_inx - search_shift - 1] - beat_avg0[p_inx - search_shift - 1]) < 0.027 \
                              and (abs_peak_max > 1 or np.abs(beat_avg0[qrs_peak_inx]) > 1)

                    # discard cases with suspected false detections (2)
                    if not (cond2_2 or cond2_3 or cond2_5):
                        log('MeasureQRSWidth P Wave.6.3.4')
                        avg_beat_features['p_inx'] = p_inx - search_shift - 1
                        avg_beat_features['p_onset'] = p_start_inx - search_shift - 1
                        avg_beat_features['pr_interval'] = (qrs_start_inx - p_start_inx) / Fs * 1e3 # ms.

                        if debug_flag:
                            plt.figure()
                            plt.plot(polarity * beat_avg0, '.-')
                            plt.plot(p_inx, polarity * beat_avg0[p_inx], 'o')
                            plt.plot(p_start_inx, polarity * beat_avg0[p_start_inx], 'o')

        # <----------------------------------------------------- P-wave location

        # v2 --->
        # low confidence on QRS measurement
        if (low_ci_flag_1 and avg_beat_features['p_inx'] == None) or low_ci_flag_2:
            log('MeasureQRSWidth Low Confidence on QRS Measurement')
            avg_beat_features['CI'] = 0  # v2 <-----


        avg_beat_features['clipped_avg_beat'] = polarity * beat_avg0
        avg_beat_features['qrs_onset'] = qrs_start_inx
        avg_beat_features['qrs_offset'] = jp
        avg_beat_features['qrs_duration'] = QRS_int

        return ([], avg_beat_features)

    except Exception as e:
        log('Exception: ', str(e))
        return ('QRS measurement error.', avg_beat_features)
