import matplotlib.pyplot as plt
import numpy as np
import scipy


def ECG_QRS_measurement(beat_avg, medianRR, Fs = 300, log=print):

    """ Look for:
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
    1) Select the maximum magnitude QRS point as the starting reference point for the search of the J-point and
    QRS onset. The initial reference point is assumed to be an R-wave (this decision can be subsequently overruled).
    2) Try to identify a potential beat polarity inversion by looking for a P-wave which is assumed to have positive
    polarity.
    3) Look for the J-point. The 1st step of the search consists in validating/overruling the initial peak selection. An
    alternative peak is selected if the initial reference is found to be an S-wave (or something else).
    The J-point search is done in an incremental way. The algorithm starts by checking the most common/simple beat morphologies.
    A negative search will lead to more complex checks of less common/abnormal morphologies. The search is mainly based on
    tests of the beat morphology using the 1st and 2nd derivatives.
    Both the tests conditions and the detection thresholds have been determined/tuned running tests on ~5000 average beats.
    4) Look for the QRS onset in a similar way as the J-point location (except for the beat polarity check).
    5) Look for the P wave and the P wave onset. This version of the algorithm considers only positive P waves.
    """

    beat_avg = np.array(beat_avg)
    debug_flag = False

    avg_beat_features = {'clipped_avg_beat': None, 'qrs_onset': None, 'qrs_offset': None, 'qrs_duration': None, 'CI': 1,
                         'p_inx': None, 'p_onset': None, 'pr_interval': None} # <---- v2

    if Fs != 300:
        return('Unsupported sampling frequency Fs.', avg_beat_features)

    # sanity checks
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

        # b, a = scipy.signal.butter(1, [4 / (Fs / 2), 60 / (Fs / 2)], btype='bandpass')
        b = np.array([ 0.39918232,  0.        , -0.39918232])
        a = np.array([ 1.        , -1.13061563,  0.20163537 ])
        # remove baseline; enhance peaks (particularly in QS morphologies where the QR waves are rather a down-slope edge)
        beat_avg_bpf = scipy.signal.filtfilt(b, a, beat_avg)

        # QRS reference peak search ---------------------------------->
        # Assumption: maximum magnitude point within the search window is an R wave (this assumption can be overruled subsequently)
        inx = 180 # qrs peak index in many average beats
        wl = 35
        if HR > 100:
            wl -= 5
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
        beat_avg = beat_avg[inxs:inxe] # miguel2, bug


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


        # look for potential distorted/noisy beats, or beats where we are not confident on an accurate measurement (unusual morphologies)
        low_ci_flag_1 = False; low_ci_flag_2 = False
        if (distor_ratio * 100 >= 4.7 and noise_ratio * 100 >= 12.6):
            log('MeasureQRSWidth quality assessment.1')
            low_ci_flag_1 = True # discard only if p-wave not found
        if (distor_ratio * 100 >= 14 and noise_ratio * 100 >= 12):
            log('MeasureQRSWidth quality assessment.2')
            low_ci_flag_2 = True # discard whether or not a p-wave is found
        if (distor_ratio * 100 >= 37 and noise_ratio * 100 >= 10) or \
           (distor_ratio * 100 >= 9 and noise_ratio * 100 >= 16) or \
           (distor_ratio * 100 >= 14.5 and noise_ratio * 100 >= 3.5 and norm_coeff < 0.27 and
           (np.max([beat_avg0[0], beat_avg0[-1]]) - np.min([beat_avg0[0], beat_avg0[-1]])) > 0.015):
            log('MeasureQRSWidth quality assessment.3')
            low_ci_flag_2 = True  # discard whether or not a p-wave is found
        if (distor_ratio * 100 >= 6.5 and noise_ratio * 100 >= 6.9 and norm_coeff > 0.5) or \
           (beat_avg0[qrs_peak_p_inx - inxs] - beat_avg0[qrs_peak_n_inx - inxs]) < 0.25 and distor_ratio * 100 > 19 or \
           (beat_avg0[qrs_peak_p_inx - inxs] - beat_avg0[qrs_peak_n_inx - inxs]) < 0.33 and distor_ratio * 100 > 35:
            # avg_beat_features['CI'] = 0 # potential low quality beat; QRS and PR measurements should be discarded
            log('MeasureQRSWidth quality assessment.4')
            low_ci_flag_1 = True # discard only if p-wave not found

        avg_beat_features['distor_ratio'] = distor_ratio
        avg_beat_features['noise_ratio'] = noise_ratio
        # < ----------------------- quality assessment of the average beat


        QS_flag = False

        # ----------------------- P-wave evidence search ---------------------------------
        # Baseline trend estimation before QRS
        x2 = p1 - inxs - 16 # try to find a sample (x2, y2) between the P-wave and the QRS
        if p1 - inxs < 63:
            x2 += 10

        # Linear estimation of the baseline before the QRS
        y2 = beat_wave_lpf_norm[x2]
        y1 = beat_wave_lpf_norm[0]
        slope = (y2 - y1) / x2
        intercept = y2 - slope * x2
        bl_est = slope * np.array(range(x2)) + intercept

        # Subtract the baseline linear trend to the beat and compute the derivative of the result.
        # 'deriv' should account for the beat slope variations before the QRS, with a mitigated baseline distortion.
        deriv = np.diff(beat_wave_lpf_norm[:x2] - bl_est)*10

        if debug_flag:
            plt.figure()
            plt.plot(beat_wave_lpf_norm, '*-')
            plt.plot(bl_est)
            plt.plot(beat_wave_lpf_norm[:x2] - bl_est)
            plt.plot(deriv, '*-')
            plt.grid(True)

        deriv_pmax_inx = np.where(deriv == np.max(deriv))[0][0]
        deriv_nmin_inx = np.where(deriv == np.min(deriv))[0][0]

        # Assuming max(deriv) and min(deriv) correspond to the up/down slopes of the P-wave, their position
        # with respect to each other indicates the polarity of the wave.
        cond6 = True
        # Try to avoid scenarios where max(deriv) and/or min(deriv) don't correspond to the P-wave. For example, when
        # the sample (x2, y2) is not between the P-wave and the QRS but rather on top of the P-wave or the QRS.
        if np.max(np.abs([deriv[deriv_pmax_inx], deriv[deriv_nmin_inx]])) > .055 \
                  and len(deriv) > 30 and np.abs(slope) < 5e-3\
                  and deriv_pmax_inx > 1\
                  and np.abs(deriv_nmin_inx - deriv_pmax_inx) < 40\
                  and np.min(np.abs([deriv[deriv_pmax_inx], deriv[deriv_nmin_inx]])) > .027\
                  and ((len(deriv) - deriv_pmax_inx) > 3)\
                  and ((len(deriv) - deriv_nmin_inx) > 3):
            # P-wave evidence found
            log('P-wave evidence.')
            if (deriv_nmin_inx < deriv_pmax_inx):
                # Inverted P-wave
                log('P-wave evidence: polarity inversion')
                qrs_peak_inx = p1 - inxs if qrs_peak_inx == p2 - inxs else p2 - inxs
                beat_wave_diff = -beat_wave_diff
                beat_wave_lpf_norm = -beat_wave_lpf_norm
                inv_polarity = not inv_polarity
                peak_test = np.where(beat_wave_lpf_norm[qrs_peak_inx - 2:qrs_peak_inx + 3] ==
                                     np.max(beat_wave_lpf_norm[qrs_peak_inx - 2:qrs_peak_inx + 3]))[0][0]
                peak_found = 0 < peak_test < 4
                if not peak_found and not (beat_wave_lpf_norm[qrs_peak_inx] > 0.62) \
                        and (qrs_peak_inx == p1 - inxs) or\
                        (qrs_peak_inx == p2 - inxs and beat_wave_lpf_norm[qrs_peak_inx] < 0.25) or \
                        (qrs_peak_inx == p2 - inxs and not peak_found): # also evidence of QS morphology
                    QS_flag = True
                    log('QS morphology evidence.')

                cond6 = False
            elif deriv_pmax_inx < deriv_nmin_inx:
                # positive polarity P-wave evidence
                cond6 = False # avoid potential polarity reinversion when looking for the J-point
        # <----------------------- P-wave evidence search ----------------------------



        # J-point location ------------------------------------------------------------------------------------>
        slope_test2 = (beat_wave_lpf_norm[p2 - inxs + 1] - beat_wave_lpf_norm[p2 - inxs]) * \
                      (beat_wave_lpf_norm[p2 - inxs] - beat_wave_lpf_norm[p2 - inxs - 1])
        if QS_flag and qrs_peak_inx == p2 - inxs and slope_test2 > 0:
            log('MeasureQRSWidth J-Point.1')
            inxTot = 0  # QS morphology; use right peak as J-point
            aux = beat_wave_diff[qrs_peak_inx + 1:]
            if np.min(aux[:8]) < -4e-2:
                log('MeasureQRSWidth J-Point.1.1')
                inx = np.where(aux[:15] > -5e-3)[0][0]
                inxTot += inx
        else:
            log('MeasureQRSWidth J-Point.2')
            thresh = 0.0113333

            aux = beat_wave_diff[qrs_peak_inx + 1:]
            aux3 = beat_wave_lpf_norm[qrs_peak_inx + 1:]

            # If the reference peak doesn't look like an R-wave (could be an inverted S or Q wave), switch to the
            # 2nd peak option.
            inx_aux = np.where(aux >= 0)[0][0]
            try:
                ndev_test = len(np.where(aux[:75] < 0)[0]) / len(aux[:75]) > 0.62
            except:
                ndev_test = False

            aux2 = np.diff(aux)  # 2nd derivative
            cond1 = inx_aux > 38 or (inx_aux > 28 and HR > 97) or ndev_test \
                    and np.min(aux[20:57]) < -0.00152 and slope_test2 < 1.35e-3
            cond2 = np.min(aux[30:110]) <= -1.84e-3
            cond3 = not (np.min(aux2[1:10]) < -1.5e-2 and np.max(aux[1:10]) < 0
                    and len(np.where(np.diff(np.sign(aux2[0:10])) < 0)[0]) > 1 and inx_aux < 55)
            cond4 = qrs_peak_inx == p2 - inxs
            cond5 = not (np.mean(aux3[35:70]) < -0.25 and np.mean(abs(aux[35:70])) < 0.005)


            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
                log('MeasureQRSWidth J-Point.2.1')
                # The main reasons for all these conditions to be true are:
                #     - an average beat with inverted polarity
                #     - an initial peak selection of an S-wave
                #     - QRS with QS morphology.

                qrs_peak_inx = p1 - inxs
                QS_flag = True  # potential QS-like morphology

                beat_wave_diff = -beat_wave_diff  # invert polarity
                aux = beat_wave_diff[qrs_peak_inx + 1:]
                beat_wave_lpf_norm = -beat_wave_lpf_norm
                inv_polarity = not inv_polarity

            if debug_flag:
                plt.figure()
                plt.plot(beat_wave_lpf_norm, '.-')
                plt.plot(qrs_peak_inx, beat_wave_lpf_norm[qrs_peak_inx], 'o')


            # Most general case. Start search from R-wave peak.
            inx0 = np.where(aux[:56] == np.min(aux[:56]))[0][0] # start search at minimum slope point

            inxTot = inx0 + 1
            aux = aux[inx0:]
            high_slope_flag = False

            # 1st pass --------------------------------
            # In most cases, it should find the S-wave, that could also be the J-point.
            inx1 = np.where(aux > 0)[0]  # derivative changes sign
            inx2 = np.where(np.abs(aux) <= thresh)[0] # derivative close to 0
            if len(inx2) == 0:  # quickly oscillating baseline
                inx2 = [np.Inf]

            inx = int(np.min([inx1[0], inx2[0]]))
            inxTot += inx
            aux = aux[inx:]
            # -----------------------------------------

            # slurring QRS
            slur_flag = False
            if (np.min(aux[2:7]) < -7.6e-3 and np.max(aux[:5]) < 0.058 and inx_aux <= 50 and norm_coeff > 0.5)\
                    or np.min(aux[:6]) < -0.015\
                    or (norm_coeff > 1.1 and np.min(aux[:7]) < -6e-3):
                log('MeasureQRSWidth J-Point.2.2')
                aux_tmp = aux
                inxTotCpy = inxTot

                inxTot += 5
                aux = aux[5:]

                inx1 = np.where(aux > 0)[0]
                if len(inx1) == 0:
                    inx1 = [np.Inf]

                if norm_coeff < 0.73:
                    thresh1 = 5e-3
                else:
                    thresh1 = 3e-3

                inx2 = np.where(np.abs(aux) < thresh1)[0]
                if len(inx2) == 0:
                    inx2 = [np.Inf]

                inx = int(np.min([inx1[0], inx2[0]]))
                if inx < 14 and inx1[0] < 51:
                    log('MeasureQRSWidth J-Point.2.2.1')
                    inxTot += inx
                    aux = aux[inx:]

                slur_flag = True

                if inx == 0: # rollback, miguel
                    log('MeasureQRSWidth J-Point.2.2.2')
                    aux = aux_tmp
                    inxTot = inxTotCpy
                    slur_flag = False


            # After S-wave
            thresh1 = 0.0065
            if np.max(aux[:9]) > thresh1 and not slur_flag and abs_peak_max > 0.17:  # additional non-negligible
                                                                                     # slope increase
                log('MeasureQRSWidth J-Point.2.3')
                aux_tmp = aux
                inxTotCpy = inxTot

                inx = np.where(np.max(aux[:9]) == aux[:9])[0][0]
                inxTot += inx + 1
                aux = aux[inx + 1:]

                inx1 = np.where(aux < 0)[0]  # derivative changes sign
                if len(inx1) == 0:
                    inx1 = [np.Inf]

                if norm_coeff < 0.35:
                    thresh1 = 0.031
                    log('MeasureQRSWidth J-Point.2.3.1 thresh1: %s' % thresh1)
                else:
                    thresh1 = 0.0184
                    log('MeasureQRSWidth J-Point.2.3.2 thresh1: %s' % thresh1)

                inx2 = np.where(np.abs(aux) <= thresh1)[0]  # derivative close to 0
                if len(inx2) == 0:
                    inx2 = [np.Inf]

                inx = int(np.min([inx1[0], inx2[0]]))

                if inx < 3:
                    log('MeasureQRSWidth J-Point.2.3.3')
                    if norm_coeff > 0.74:
                        thresh1 = 1e-2
                        log('MeasureQRSWidth J-Point.2.3.3.1 thresh1: %s' % thresh1)
                    else:
                        thresh1 = 1e-2 # miguel 5e-3 -> 7e-3 -> 1e-2
                        log('MeasureQRSWidth J-Point.2.3.3.2 thresh1: %s' % thresh1)
                    inx = np.where(np.abs(aux) <= thresh1)[0]
                    if len(inx) == 0:
                        inx = 0
                    else:
                        inx = inx[0]

                if inx > 13:  # restart search with a higher detection threshold; overrule previous inx
                    log('MeasureQRSWidth J-Point.2.3.4')
                    high_slope_flag = True
                    inx = np.where(np.abs(aux) <= 2.1e-2)[0]
                    if len(inx) == 0:
                        inx = np.Inf
                    else:
                        inx = inx[0]

                    if inx > 10:  # restart search with a higher detection threshold; overrule previous inx
                        log('MeasureQRSWidth J-Point.2.3.4.1')
                        high_slope_flag = True
                        inx = np.where(np.abs(aux) <= 1e-1)[0]
                        if len(inx) == 0:
                            inx = np.Inf
                        else:
                            inx = inx[0]

                inxTot += inx
                aux = aux[inx:]

                if (inx < 2 and high_slope_flag) or (np.min(aux[:10]) == aux[0] and not slur_flag):  # rollback
                    log('MeasureQRSWidth J-Point.2.3.5')
                    high_slope_flag = False
                    aux = aux_tmp
                    inxTot = inxTotCpy


            # multiple types of wide QRS morphologies...

            # wide QRS (1)
            if np.min(aux[6:15]) < -4.5e-2:
                log('MeasureQRSWidth J-Point.2.4')
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
                    log('MeasureQRSWidth J-Point.2.4.1')
                    inxTot += inx
                    aux = aux[inx:]

            if norm_coeff >= 0.19:
                thresh2 = -0.0245
                log('MeasureQRSWidth J-Point.2.5 thresh2: %s' % thresh2)
            else:
                thresh2 = -0.0105
                log('MeasureQRSWidth J-Point.2.6 thresh2: %s' % thresh2)

            # wide QRS (2)
            log('MeasureQRSWidth J-Point.2 np.max(aux[8:21]): %s' % np.max(aux[8:21]))
            log('MeasureQRSWidth J-Point.2 np.min(aux[20:50]): %s' % np.min(aux[20:50]))
            log('MeasureQRSWidth J-Point.2 high_slope_flag: %s' % high_slope_flag)
            log('MeasureQRSWidth J-Point.2 np.max(aux[:8]): %s' % np.max(aux[:8]))


            if (HR < 115 and np.max(aux[9:21]) > 2.2e-2 and np.min(aux[20:50]) > thresh2 and not high_slope_flag) or \
                    (np.max(aux[:8] > 9.3e-2) and HR < 134):
                log('MeasureQRSWidth J-Point.2.7')

                aux_tmp = aux
                inxTotCpy = inxTot
                inx = np.where(aux[8:20] == np.max(aux[8:20]))[0][0]
                inxTot += inx + 8
                aux = aux[inx + 8:]

                if QS_flag and aux[0] < 7.6e-2:
                    thresh2 = 7.5e-3
                    log('MeasureQRSWidth J-Point.2.7.1 thresh2: %s' % thresh2)
                elif np.max(aux_tmp[:8]) > 8e-2:
                    thresh2 = 0.02
                    log('MeasureQRSWidth J-Point.2.7.2 thresh2: %s' % thresh2)
                else:
                    thresh2 = 0.00375
                    log('MeasureQRSWidth J-Point.2.7.3 thresh2: %s' % thresh2)

                inx1 = np.where(np.abs(aux) <= thresh2)[0]
                inx2 = np.where(aux < 0)[0]
                if len(inx2) == 0:
                    inx2 = [np.Inf]
                inx = int(np.min([inx1[0], inx2[0]]))
                inxTot += inx

                aux = aux[inx:]

                if inx >= 23 or (inx >= 19 and QS_flag) or inx < 2:  # rollback
                    log('MeasureQRSWidth J-Point.2.7.4')
                    aux = aux_tmp
                    inxTot = inxTotCpy

            # wide QRS (3)
            elif np.min(aux[:10]) < -3e-2 and np.max(aux[10:20]) < 0.075 and inx_aux < 64 and np.where(aux[1:] > 0)[0][0] < 36:
                log('MeasureQRSWidth J-Point.2.8')
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


            # wide QRS (4)
            if np.max(aux[:25]) > 0.0425 or \
                     (np.max(aux[:25]) > 0.02 and slur_flag ) or \
                      np.max(aux[:25]) > 0.03 and QS_flag:
                inx1 = np.where(np.max(aux[:20]) == aux[:20])[0][0]
                log('MeasureQRSWidth J-Point.2.9 inx1: %s' % inx1)
                aux_tmp = aux[inx1: inx1 + 20]
                inx = np.where(np.abs(aux_tmp) <= aux[inx1]/4)[0]
                if len(inx) > 0:
                    log('MeasureQRSWidth J-Point.2.9.1 inx: %s' % inx[0])
                    inx = inx[0] + inx1
                    inxTot += inx


        jp = qrs_peak_inx + 1 + inxTot
        log('MeasureQRSWidth J-Point jp: %s' % jp)
        # <------------------------------------------ J-point location

        if debug_flag:
            plt.plot(jp, beat_wave_lpf_norm[jp], 'o')

        # ---------------------------------------- QRS onset ----------------------------------------
        aux = beat_wave_diff[:qrs_peak_inx - 1]
        aux = np.array(aux)
        aux = -aux[::-1]
        log('MeasureQRSWidth QRS onset  norm_coeff: %s' % norm_coeff)

        # miguel3
        inx = np.where(aux > -8e-3)[0]
        if not QS_flag and len(inx) == 0:
            log('MeasureQRSWidth QRS onset.0')
            low_ci_flag_2 = True  # discard measurement; no derivative >=0 to the left of the QRS

        if np.min(aux[:20]) <= -1.1e-2 and QS_flag and not (qrs_peak_inx == p2 - inxs) and\
                ((beat_wave_lpf_norm[qrs_peak_inx] - beat_wave_lpf_norm[qrs_peak_inx - 2]) > -1.2e-2):
            # overrule QS-like morphology detection
            log('MeasureQRSWidth QRS onset.1')
            norm_coeff = np.abs(beat_avg[qrs_peak_inx])
            QS_flag = False

        if QS_flag and np.min(aux[:3]) > -0.038 and qrs_peak_inx == p1 - inxs: # QS-like morphology
            inxTot = 0
            log('MeasureQRSWidth QRS onset.2')

            if beat_wave_lpf_norm[qrs_peak_inx] > 0:
                inxTot = -np.where(beat_wave_lpf_norm[qrs_peak_inx:] <= 0)[0][0]
                log('MeasureQRSWidth QRS onset.2.1')

        else:  # at this point, in most cases the search should be starting from an R-wave peak
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

            inxTot = inx0 + 1
            aux = aux[inx0:]

            # 1st pass
            aux2 = np.diff(aux) # 2nd derivative
            inx1 = np.where(aux > 0)[0]  # derivative changes sign
            if len(inx1) == 0:
                inx1 = [np.Inf]

            if norm_coeff < 0.45:
                thresh1 = 0.038
                log('MeasureQRSWidth QRS onset.3.2 thresh1: %s' % thresh1)
            else:
                thresh1 = 0.01
                log('MeasureQRSWidth QRS onset.3.3 thresh1: %s' % thresh1)

            inx2 = np.where(np.abs(aux) <= thresh1)[0]
            if len(inx2) == 0:
                inx2 = [np.Inf]

            inx3 = np.where(aux2 < 5e-4)[0]
            # if len(inx3) < 3:
            if len(inx3) < 3 or not QS_flag:
                inx3 = [np.Inf]
            else:
                inx3 = inx3[inx3 > 8]

            inx = int(np.min([inx1[0], inx2[0], inx3[0]]))
            inxTot += inx
            aux = aux[inx:]  # QRS complex with no Q deflection

            if norm_coeff > 2.5:
                thresh1 = 7.5e-3
            else:
                thresh1 = 0.0055

            # 2nd pass
            if np.max(aux[:12]) >= thresh1 and np.min(aux[1:10]) > -5e-2:  # look for Q deflection
                log('MeasureQRSWidth QRS onset.3.5')
                aux_tmp = aux
                inx1 = np.where(aux[:12] == np.max(aux[:12]))[0][0]
                aux = aux[inx1:]
                aux2 = np.diff(aux)

                if norm_coeff < 0.47:
                    thresh1 = 6e-3
                else:
                    thresh1 = 2.8e-3

                inx2 = np.where(np.abs(aux) <= thresh1)[0]
                if len(inx2) == 0:
                    inx2 = [np.Inf]

                inx3 = np.where(aux < 0)[0]
                if len(inx3) == 0:
                    inx3 = [np.Inf]

                inx4 = np.where(aux2 >= 0)[0]
                if len(inx4) > 0 and np.min(inx4) >= 4 and not QS_flag:
                    log('MeasureQRSWidth QRS onset.3.6')
                    inx4 = inx4[inx4 >= 4][0] - 1
                else:
                    log('MeasureQRSWidth QRS onset.3.7')
                    inx4 = np.Inf

                inx5 = int(np.min([inx2[0], inx3[0], inx4]))
                inx = inx1 + inx5

                if (inx < 19 and np.max(aux[inx5 + 1:inx5 + 1 + 5]) < 0.0125 and
                        np.min(aux[inx5 + 4:inx5 + 15]) > -4.4e-3 and inx5 > 1) or \
                        (beat_wave_lpf_norm[qrs_peak_inx] > 1.5 and np.min(aux[inx5:inx5 + 17]) > -0.013) or\
                        (inx <= 19 and inx5 == inx4 and np.max(aux[10:20]) > 0.015) or \
                        (inx <= 19 and inx5 == inx4 and np.min(aux[inx5 + 4:inx5 + 16]) > -6.25e-3):
                    log('MeasureQRSWidth QRS onset.3.8')
                    inxTot += inx - 2
                else: # rollback
                    log('MeasureQRSWidth QRS onset.3.9')
                    aux = aux_tmp
            elif np.min(aux[1:12]) < -0.0174 and np.max(aux[5:15]) < -2e-3 and norm_coeff > 0.4:  # slurry QRS
                log('MeasureQRSWidth QRS onset.4')
                aux_tmp = aux
                inxTotCpy = inxTot

                inx = np.where(np.abs(aux) <= 6e-3)[0]
                inx = inx[inx > 5]
                inxTot += inx[0]
                # 2nd pass
                aux = aux[inx[0]:]
                inx = np.where(aux >= 0)[0]

                if len(inx) == 0:
                    inx = [np.Inf]
                if inx[0] > 19 and np.min(aux[:20]) > -9.3e-3:
                    log('MeasureQRSWidth QRS onset.4.1')
                    aux = aux_tmp
                    inxTot = inxTotCpy

            # multiwave or QS-like morphologies
            if (aux[0] > 0.17 and np.min(aux[:15]) < -0.11) or \
                    (aux[0] > 0.17 and np.min(aux[:15]) < -0.05 and qrs_peak_inx == (p2 - inxs)) or \
                    (aux[0] > 0.05 and np.min(aux[:15]) < -0.079) or\
                    (aux[0] > 0.06 and np.min(aux[:15]) < -0.035) or\
                    (np.min(aux[:15]) < -0.2 and np.max(aux[:15]) > -0.12 and qrs_peak_inx == (p2 - inxs)) or \
                    (QS_flag and qrs_peak_inx == p1 - inxs) or \
                    (QS_flag and np.min(aux[:20]) < -1.4e-2) or \
                    (np.min(aux[:20]) < -0.15 and np.max(aux[:20]) < 0.04):
                log('MeasureQRSWidth QRS onset.5')
                inx0 = np.where(aux[:15] == np.min(aux[:15]))[0][0]
                inx1 = np.where(np.abs(aux[inx0:]) < 1e-2)[0][0]

                if inx0 + inx1 < 36:
                    log('MeasureQRSWidth QRS onset.5.1')
                    inxTot += inx0 + inx1
                    aux = aux[inx0 + inx1:]
                    if np.max(aux[:6]) > 0.098: # miguel, bug
                        log('MeasureQRSWidth QRS onset.5.1.1')
                        inx0 = np.where(aux[:6] == np.max(aux[:6]))[0][0]
                        inx1 = np.where(np.abs(aux[inx0:]) < 1e-2)[0][0]
                        inxTot += inx0 + inx1

            # Wide Q-wave
            if (np.max(aux[5:20]) > 8.6e-2) and np.min(aux[25:40]) > -1e-3:
                log('MeasureQRSWidth QRS onset.6')
                inx0 = np.where(aux[5:20] == np.max(aux[5:20]))[0][0]
                inx1 = np.where(aux[inx0 + 5:] < 1e-2)[0][0]
                if inx1 != len(aux):
                    log('MeasureQRSWidth QRS onset.6.1')
                    aux = aux[inx0 + inx1 + 5:]
                    inxTot += inx0 + inx1 + 5
                    if np.max(aux[5:15]) > 0.028 and np.min(aux[5:25]) > -2.5e-3:
                        log('MeasureQRSWidth QRS onset.6.1.1')
                        inx0 = np.where(aux[:15] == np.max(aux[0:15]))[0][0]
                        inx1 = np.where(np.abs(aux[inx0:inx0 + 10]) <= 4e-3)[0][0]
                        inxTot += inx0 + inx1


        qrs_start_inx = qrs_peak_inx - 1 - inxTot
        log('MeasureQRSWidth QRS onset qrs_start_inx: %s' % qrs_start_inx)

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

        if debug_flag:
            plt.figure()
            plt.plot(polarity * beat_avg0, '.-')
            plt.plot(aux, '.-')

        cond_det = (len(peaks) + len(peaks2)) > 0
        cond_det_4 = not (len(peaks) >= 3)
        cond_det_5 = not (len(peaks) == 0 and not negative_p_det)


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
                cond6 = (np.max(peaks_properties['prominences']) / np.min(peaks_properties['prominences']) < 1.45 )
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

        # low confidence on QRS measurement
        if (low_ci_flag_1 and avg_beat_features['p_inx'] == None) or low_ci_flag_2:
            log('MeasureQRSWidth Low Confidence on QRS Measurement')
            avg_beat_features['CI'] = 0


        avg_beat_features['clipped_avg_beat'] = polarity * beat_avg0
        avg_beat_features['qrs_onset'] = qrs_start_inx
        avg_beat_features['qrs_offset'] = jp
        avg_beat_features['qrs_duration'] = QRS_int
        avg_beat_features['inxs'] = inxs  # for plots outside the function with reference QRS onset/offset annotations

        return ([], avg_beat_features)

    except Exception as e:
        log('Exception: ', str(e))
        return ('QRS measurement error.', avg_beat_features)
