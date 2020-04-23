import matplotlib.pyplot as plt
import numpy as np
import scipy


def ECG_QRS_measurement(beat_avg, medianRR, Fs=300, log=print):

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
    1) Select the maximum magnitude QRS point as the starting reference point for the search of the J-point and QRS
    onset. The initial reference point is assumed to be an R-wave (this decision can be subsequently overruled).
    2) Try to identify a potential beat polarity inversion by looking for a P-wave which is assumed to have positive
    polarity.
    3) Look for the J-point. The 1st step of the search consists in validating/overruling the initial peak selection. An
    alternative peak is selected if the initial reference is found to be an S-wave (or something else).
    The J-point search is done in an incremental way. The algorithm starts by checking the most common/simple beat
    morphologies. A negative search will lead to more complex checks of less common morphologies. The search
    is mainly based on tests of the beat morphology using the 1st and 2nd derivatives.
    Both the tests conditions and the detection thresholds have been determined running tests on ~10000 average
    beats.
    4) Look for the QRS onset in a similar way as the J-point location (except for the beat polarity check).
    5) Look for the P wave peak and the P wave onset. This version of the algorithm considers only positive P waves.
    """

    beat_avg = np.array(beat_avg)
    debug_flag = False

    avg_beat_features = {'clipped_avg_beat': None, 'qrs_onset': None, 'qrs_offset': None, 'qrs_duration': None, 'CI': 1,
                         'p_inx': None, 'p_onset': None, 'pr_interval': None}

    if Fs != 300:
        return'Unsupported sampling frequency Fs.', avg_beat_features

    # sanity checks
    if medianRR == 0:
        return 'Abnormal HR.', avg_beat_features
    HR = 1 / (medianRR * 1e-3) * 60  # bpm
    log('MeasureQRSWidth heart_rate: %s' % HR)
    if HR > 201:
        return 'Abnormal HR.', avg_beat_features
    # elif: others to be added...

    medianRR = medianRR * 1e-3 * Fs  # ms -> nb samples
    log('MeasureQRSWidth median_rr_samples: %s' % medianRR)

    try:
        beat_avg0 = beat_avg
        beat_avg = beat_avg - np.mean(beat_avg)

        avg_beat_features['clipped_avg_beat'] = beat_avg0 # already defined in case of returned error

        # b, a = scipy.signal.butter(1, [4 / (Fs / 2), 60 / (Fs / 2)], btype='bandpass')
        b = np.array([0.39918232,  0.        , -0.39918232])
        a = np.array([1.        , -1.13061563,  0.20163537])
        # remove baseline; enhance peaks (particularly in QS morphologies)
        beat_avg_bpf = scipy.signal.filtfilt(b, a, beat_avg)

        # QRS reference peak search ---------------------------------->
        # Assumption: maximum magnitude point within the search window is an R wave
        # (this assumption can be overruled subsequently)
        inx = 180  # qrs peak index in many average beats
        wl = 35
        if HR > 100:
            wl -= 5
        wrange = np.arange(inx - wl, inx + wl)

        qrs_peak_inx = np.where(np.abs(beat_avg_bpf[wrange]) == np.max(np.abs(beat_avg_bpf[wrange])))[0]
        abs_peak_max = np.abs(beat_avg_bpf[wrange][qrs_peak_inx])
        cond1 = len(qrs_peak_inx) > 1  # more than 1 peak
        cond2 = np.max(beat_avg0[wrange]) < -0.2  # peak at unexpected location
        # other conditions...

        if cond1 or cond2:
            return ('QRS measurement error.', avg_beat_features)
        else:
            qrs_peak_inx = qrs_peak_inx[0]

        qrs_peak_p_inx = np.where(beat_avg_bpf[wrange] == np.max(beat_avg_bpf[wrange]))[0][0]  # positive peak
        qrs_peak_n_inx = np.where(beat_avg_bpf[wrange] == np.min(beat_avg_bpf[wrange]))[0][0]  # negative peak

        qrs_peak_inx = int(inx - wl + qrs_peak_inx)
        qrs_peak_p_inx = int(inx - wl + qrs_peak_p_inx)
        qrs_peak_n_inx = int(inx - wl + qrs_peak_n_inx)

        p1 = np.min([qrs_peak_p_inx, qrs_peak_n_inx])  # leftmost peak
        p2 = np.max([qrs_peak_p_inx, qrs_peak_n_inx])  # rightmost peak

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
        beat_avg = beat_avg[inxs:inxe]


        # quality assessment of the average beat ------------------->
        # 'distor_ratio': baseline variation w.r.t. the QRS estimated amplitude
        # 'noise_var': average beat 'smoothness' estimation
        distor_ratio = np.abs(beat_avg0[0] - beat_avg0[-1]) / \
                        (beat_avg0[qrs_peak_p_inx - inxs] - beat_avg0[qrs_peak_n_inx - inxs])
        log('MeasureQRSWidth quality assessment distort_ratio: %s' % distor_ratio)

        if debug_flag:
            print('distor ratio : {:.2f}'.format(distor_ratio*100))

        try:
            diff_beat_avg0_aux = np.diff(beat_avg0)
            diff_beat_avg0_aux[np.abs(diff_beat_avg0_aux) < 6e-4] = 0.0
            noise_var = len(np.where(np.diff(np.sign(diff_beat_avg0_aux)) != 0)[0])
            noise_ratio = noise_var / len(beat_avg0)
            if debug_flag:
                print('noise ratio: {:.2f}'.format(noise_ratio*100))
        except:
            noise_ratio = 0
            if debug_flag:
                print('noise ratio: {:.2f}'.format(noise_ratio*100))
        log('MeasureQRSWidth quality assessment noise_ratio: %s' % noise_ratio)


        # look for potential distorted/noisy beats, or beats where we are not confident on an accurate
        # measurement (unusual morphologies)
        low_ci_flag_1 = False; low_ci_flag_2 = False
        if distor_ratio * 100 > 59 or\
           (noise_ratio * 100 > 10 and norm_coeff < 0.065) or \
           (distor_ratio * 100 > 8 and noise_ratio * 100 > 18 and norm_coeff < 0.2) or \
           (distor_ratio * 100 >= 4 and noise_ratio * 100 >= 7 and norm_coeff < 0.05) or \
           (distor_ratio * 100 >= 27 and noise_ratio * 100 >= 10 and norm_coeff < 0.17):
            log('MeasureQRSWidth quality assessment.1')
            low_ci_flag_2 = True  # discard whether or not a p-wave is found
        if (distor_ratio * 100 >= 14.7 and noise_ratio * 100 >= 11 and norm_coeff < 0.35) or \
           (distor_ratio * 100 >= 30 and noise_ratio * 100 >= 9 and norm_coeff < 0.2) or \
           (noise_ratio * 100 >= 12 and norm_coeff < 0.11) or \
           (distor_ratio * 100 >= 6 and noise_ratio * 100 >= 16 and norm_coeff < 0.2):
            log('MeasureQRSWidth quality assessment.2')
            low_ci_flag_1 = True  # discard only if p-wave not found

        avg_beat_features['distor_ratio'] = distor_ratio
        avg_beat_features['noise_ratio'] = noise_ratio
        # < ----------------------- quality assessment of the average beat


        QS_flag = False

        # ----------------------- P-wave evidence search --------------------------------->
        # Linear baseline trend estimation before QRS
        x2 = p1 - inxs - 16  # try to find a sample (x2, y2) between the P-wave and the QRS
        if p1 - inxs < 63:
            x2 += 10

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

        # Assuming max(deriv) and min(deriv) correspond to the up/down slopes of the P-wave, their position
        # with respect to each other indicates the polarity of the wave.
        deriv_pmax_inx = np.where(deriv[:-8] == np.max(deriv[:-8]))[0][0]
        deriv_nmin_inx = np.where(deriv[:-4] == np.min(deriv[:-4]))[0][0]

        # If the baseline before the QRS seems to oscillate, don't look for P-wave evidence.
        # Potential oscillations are detected by looking for high amplitude derivatives on the baseline, other than
        # 'deriv_pmax_inx' and 'deriv_nmin_inx'.
        wavy_baseline_flag = False
        if deriv_pmax_inx < (len(deriv) - 20):
            inx = np.where(deriv[deriv_pmax_inx + 10:-8] == np.max(deriv[deriv_pmax_inx + 10:-8]))[0][0]
            if inx > 4 and deriv[deriv_pmax_inx + 10 + inx]/deriv[deriv_pmax_inx] > 0.71:
                wavy_baseline_flag = True
        elif 10 < deriv_pmax_inx:
            inx = np.where(deriv[:deriv_pmax_inx - 9] == np.max(deriv[:deriv_pmax_inx - 9]))[0][0]
            if 0 < deriv[deriv_pmax_inx] / deriv[inx] < 1.55 and (deriv_pmax_inx - inx) > 20:
                wavy_baseline_flag = True
        elif 2 < deriv_nmin_inx < (len(deriv) - 10 - 4):
            inx = np.where(deriv[deriv_nmin_inx + 10:-4] == np.min(deriv[deriv_nmin_inx + 10:-4]))[0][0]
            if inx > 10 and deriv[deriv_nmin_inx + 10 + inx] / deriv[deriv_nmin_inx] > 0.61:
                wavy_baseline_flag = True
            else:
                inx = np.where(deriv[:np.max([1, deriv_nmin_inx - 4])] ==
                               np.min(deriv[:np.max([1, deriv_nmin_inx - 4])]))[0][0]
                if (deriv_nmin_inx - inx) > 24 and deriv[inx] / deriv[deriv_nmin_inx] > 0.53:
                    wavy_baseline_flag = True


        cond6 = True
        # Try to avoid scenarios where max(deriv) and/or min(deriv) don't correspond to P-wave slopes. For example, when
        # the sample (x2, y2) is not between the P-wave and the QRS but rather on the P-wave or the QRS.
        if not wavy_baseline_flag \
           and np.max(np.abs([deriv[deriv_pmax_inx], deriv[deriv_nmin_inx]])) > .0395 \
           and len(deriv) > 30 and np.abs(slope) < 4.5e-3 \
           and deriv_pmax_inx > 1 \
           and deriv_nmin_inx > 1 \
           and np.abs(deriv_nmin_inx - deriv_pmax_inx) < 38 \
           and np.abs(deriv_nmin_inx - deriv_pmax_inx)/len(deriv) < .39 \
           and np.min(np.abs([deriv[deriv_pmax_inx], deriv[deriv_nmin_inx]])) > .027 \
           and not (deriv[deriv_pmax_inx - 1] < 0 and deriv[deriv_pmax_inx + 1] < 0) \
           and not (noise_ratio*100 > 13 and norm_coeff < 0.12):
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
                if (not peak_found and not (beat_wave_lpf_norm[qrs_peak_inx] > 0.62)
                        and (qrs_peak_inx == p1 - inxs)) or\
                        ((qrs_peak_inx == p2 - inxs) and beat_wave_lpf_norm[qrs_peak_inx] < 0.27) or \
                        ((qrs_peak_inx == p2 - inxs) and not peak_found):  # also evidence of QS morphology
                    QS_flag = True
                    log('QS morphology evidence.')

                cond6 = False
            elif deriv_pmax_inx < deriv_nmin_inx:
                # positive polarity P-wave evidence
                cond6 = False  # avoid potential polarity reinversion when looking for the J-point
        # <----------------------- P-wave evidence search ----------------------------


        # J-point location ------------------------------------------------------------------------------------>
        slope_test2 = (beat_wave_lpf_norm[p2 - inxs + 1] - beat_wave_lpf_norm[p2 - inxs]) * \
                      (beat_wave_lpf_norm[p2 - inxs] - beat_wave_lpf_norm[p2 - inxs - 1])
        if QS_flag and qrs_peak_inx == p2 - inxs and slope_test2 > 0:
            log('MeasureQRSWidth J-Point.1')
            inxTot = 0  # QS morphology; use right peak as J-point
            aux = beat_wave_diff[qrs_peak_inx + 1:]
            if np.min(aux[:8]) < -3e-2:
                log('MeasureQRSWidth J-Point.1.1')
                inx1 = np.where(aux[:8] == np.min(aux[:8]))[0][0]
                inx = np.where(np.abs(aux[inx1:15]) < 1e-2)[0][0]
                inxTot += inx1 + inx
        else:
            log('MeasureQRSWidth J-Point.2')
            thresh = 0.0113333 if norm_coeff < 1.2 else 2e-3

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
            cond1 = (inx_aux > 38 or (inx_aux > 28 and HR > 97) or ndev_test) \
                    and np.min(aux[20:57]) < -0.00152 and slope_test2 < 1.35e-3
            cond2 = np.min(aux[30:110]) <= -1.84e-3
            cond3 = not (np.min(aux2[1:10]) < -1.5e-2 and np.max(aux[1:10]) < 0
                    and len(np.where(np.diff(np.sign(aux2[0:10])) < 0)[0]) > 1 and inx_aux < 55)
            cond5 = not (np.mean(aux3[35:70]) < -0.25 and np.mean(abs(aux[35:70])) < 0.005)


            if cond1 and cond2 and cond3 and cond5 and cond6:
                log('MeasureQRSWidth J-Point.2.1')
                # The main reasons for all these conditions to be true are:
                #     - an average beat with inverted polarity
                #     - an initial peak selection of an S-wave
                #     - QRS with QS morphology.

                qrs_peak_inx = p1 - inxs if qrs_peak_inx == p2 - inxs else p2 - inxs

                beat_wave_diff = -beat_wave_diff  # invert polarity
                aux = beat_wave_diff[qrs_peak_inx + 1:]
                beat_wave_lpf_norm = -beat_wave_lpf_norm
                inv_polarity = not inv_polarity
                slope_test = (beat_wave_lpf_norm[qrs_peak_inx + 1] - beat_wave_lpf_norm[qrs_peak_inx]) * \
                             (beat_wave_lpf_norm[qrs_peak_inx] - beat_wave_lpf_norm[qrs_peak_inx - 1])
                if beat_wave_lpf_norm[qrs_peak_inx] < 0.19 or slope_test > 0:
                    QS_flag = True
                    log('MeasureQRSWidth J-Point.2.1.1')

            if debug_flag:
                plt.figure()
                plt.plot(beat_wave_lpf_norm, '.-')
                plt.plot(qrs_peak_inx, beat_wave_lpf_norm[qrs_peak_inx], 'o')


            if QS_flag and qrs_peak_inx == p2 - inxs:
                inxTot = 0
                log('MeasureQRSWidth J-Point.2.2')
                if aux[0] > 0:
                    inx = np.where(np.abs(aux[:10]) < 2.5e-3)[0]
                    log('MeasureQRSWidth J-Point.2.2.1')
                    if inx.size > 0:
                        log('MeasureQRSWidth J-Point.2.2.1.1')
                        inxTot += np.max([inx[0] - 1, 0])
            else:
                log('MeasureQRSWidth J-Point.2.3')
                # Most general case. Start search from R-wave peak.
                inx0 = np.where(aux[:56] == np.min(aux[:56]))[0][0]  # start search at minimum slope point

                inxTot = inx0 + 1
                aux = aux[inx0:]
                high_slope_flag = False

                # 1st pass --------------------------------
                # In most cases, it should find the S-wave, that could also be the J-point.
                inx1 = np.where(aux > 0)[0]  # derivative changes sign
                inx2 = np.where(np.abs(aux) <= thresh)[0]  # derivative close to 0
                if len(inx2) == 0:  # quickly oscillating baseline
                    inx2 = [np.Inf]

                inx = int(np.min([inx1[0], inx2[0]]))
                inxTot += inx
                aux = aux[inx:]
                # -----------------------------------------

                # Slurring or notched QRS.
                slur_flag = False
                if (np.min(aux[2:7]) < -7.6e-3 and np.max(aux[:5]) < 0.058 and inx_aux <= 50 and norm_coeff > .6) \
                    or np.min(aux[:6]) < -0.015\
                    or (norm_coeff > 1.14 and np.min(aux[:7]) < -6e-3):
                    log('MeasureQRSWidth J-Point.2.3.1')
                    aux_tmp = aux
                    inxTotCpy = inxTot

                    inxTot += 5
                    aux = aux[5:]

                    inx1 = np.where(aux > 0)[0]
                    if len(inx1) == 0:
                        inx1 = [np.Inf]

                    thresh1 = 5e-3 if norm_coeff < 0.73 else 3e-3

                    inx2 = np.where(np.abs(aux) < thresh1)[0]
                    if len(inx2) == 0:
                        inx2 = [np.Inf]

                    inx = int(np.min([inx1[0], inx2[0]]))
                    if inx < 14 and inx1[0] < 51:
                        log('MeasureQRSWidth J-Point.2.3.1.1')
                        inxTot += inx
                        aux = aux[inx:]

                    slur_flag = True

                    if inx == 0:  # rollback
                        log('MeasureQRSWidth J-Point.2.3.1.2')
                        aux = aux_tmp
                        inxTot = inxTotCpy
                        slur_flag = False


                # After S-wave
                if norm_coeff < 0.97:
                    thresh1 = 0.0088
                elif norm_coeff < 1.24:
                    thresh1 = 4.6e-3
                else:
                    thresh1 = 4e-3

                if np.max(aux[:9]) > thresh1 and not slur_flag and abs_peak_max > 0.15:  # additional non-negligible
                                                                                         # slope increase
                    log('MeasureQRSWidth J-Point.2.3.2')
                    aux_tmp = aux
                    inxTotCpy = inxTot

                    inx = np.where(np.max(aux[:9]) == aux[:9])[0][0]
                    inxTot += inx + 1
                    aux = aux[inx + 1:]

                    inx1 = np.where(aux < 0)[0]  # derivative changes sign
                    if len(inx1) == 0:
                        inx1 = [np.Inf]

                    thresh1 = 0.031 if norm_coeff < 0.28 else 0.013

                    inx2 = np.where(np.abs(aux) <= thresh1)[0]  # derivative close to 0
                    if len(inx2) == 0:
                        inx2 = [np.Inf]

                    inx = int(np.min([inx1[0], inx2[0]]))

                    if inx < 2 and norm_coeff > 0.2:
                        thresh1 = .5 * np.max(aux[:9])
                        log('MeasureQRSWidth J-Point.2.3.2.1')
                        inx = np.where(np.abs(aux) <= thresh1)[0]
                        if len(inx) == 0:
                            inx = 0
                        else:
                            inx = inx[0]
                        inx2 = np.where(aux < 0)[0]
                        if len(inx2) == 0:
                            inx2 = 0
                        else:
                            inx2 = inx2[0]
                        inx = np.min([inx, inx2])

                    inx_add = 0 if HR > 60 else 2
                    if inx > 13 + inx_add:
                        log('MeasureQRSWidth J-Point.2.3.2.2')
                        high_slope_flag = True
                        inx = np.where(np.abs(aux) <= 2.1e-2)[0]
                        if len(inx) == 0:
                            inx = np.Inf
                        else:
                            inx = inx[0]

                        if inx > 13:  # restart search with a higher detection threshold; overrule previous inx
                            log('MeasureQRSWidth J-Point.2.3.2.2.1')
                            # high_slope_flag = True
                            inx = np.where(np.abs(aux) <= 1e-1)[0]
                            if len(inx) == 0:
                                inx = np.Inf
                            else:
                                inx = inx[0]

                    inxTot += inx
                    aux = aux[inx:]


                    if (inx < 3 and high_slope_flag) or (np.min(aux[:10]) == aux[0] and aux[0] >= 0.01 and not slur_flag):
                        log('MeasureQRSWidth J-Point.2.3.2.3')
                        high_slope_flag = False
                        aux = aux_tmp
                        inxTot = inxTotCpy


                # Multiple types of wide QRS morphologies...

                # wide QRS (1)
                thresh1 = -4.5e-2 if norm_coeff < 0.25 else -3e-2
                if np.min(aux[6:15]) < thresh1:
                    log('MeasureQRSWidth J-Point.2.3.3')
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
                        log('MeasureQRSWidth J-Point.2.3.3.1')
                        inxTot += inx
                        aux = aux[inx:]

                thresh2 = -0.0245 if norm_coeff >= 0.19 else -0.0105

                # wide QRS (2)
                log('MeasureQRSWidth J-Point.2 np.max(aux[8:21]): %s' % np.max(aux[8:21]))
                log('MeasureQRSWidth J-Point.2 np.min(aux[20:50]): %s' % np.min(aux[20:50]))
                log('MeasureQRSWidth J-Point.2 high_slope_flag: %s' % high_slope_flag)
                log('MeasureQRSWidth J-Point.2 np.max(aux[:8]): %s' % np.max(aux[:8]))

                if (HR < 115 and np.max(aux[8:21]) > 2.4e-2 and np.min(aux[20:50]) > thresh2
                    and not (high_slope_flag or slur_flag)) or \
                    (np.max(aux[:8] > 9.3e-2) and HR < 134):
                    log('MeasureQRSWidth J-Point.2.3.4')

                    aux_tmp = aux
                    inxTotCpy = inxTot
                    inx = np.where(aux[8:20] == np.max(aux[8:20]))[0][0]
                    inxTot += inx + 8
                    aux = aux[inx + 8:]

                    if QS_flag and aux[0] < 7.6e-2:
                        thresh2 = 7.5e-3
                        log('MeasureQRSWidth J-Point.2.3.4.1 thresh2: %s' % thresh2)
                    elif np.max(aux_tmp[:8]) > 8e-2:
                        thresh2 = 0.02
                        log('MeasureQRSWidth J-Point.2.3.4.2 thresh2: %s' % thresh2)
                    else:
                        thresh2 = 0.00375
                        log('MeasureQRSWidth J-Point.2.3.4.3 thresh2: %s' % thresh2)

                    inx1 = np.where(np.abs(aux) <= thresh2)[0]
                    inx2 = np.where(aux < 0)[0]
                    if len(inx2) == 0:
                        inx2 = [np.Inf]
                    inx = int(np.min([inx1[0], inx2[0]]))
                    inxTot += inx
                    aux = aux[inx:]

                    if inx >= 23 or (inx >= 19 and QS_flag) or inx < 2:  # rollback
                        log('MeasureQRSWidth J-Point.2.3.4.4')
                        aux = aux_tmp
                        inxTot = inxTotCpy

                # wide QRS (3)
                elif np.min(aux[:10]) < -6.8e-3 and np.max(aux[10:20]) < 0.032 and inx_aux < 64 \
                                                and np.where(aux[1:] > 0)[0][0] < 36\
                                                and norm_coeff > 0.51:
                    log('MeasureQRSWidth J-Point.2.3.5')
                    inx = np.where(aux[:10] == np.min(aux[:10]))[0][0]
                    inxTot += inx
                    aux = aux[inx:]

                    inx1 = np.where(aux > 0)[0]  # derivative changes sign
                    if len(inx1) == 0:
                        inx1 = [np.Inf]

                    inx2 = np.where(np.abs(aux) <= 0.01)[0]

                    if len(inx2) == 0:
                        inx2 = [np.Inf]

                    inx = int(np.min([inx1[0], inx2[0]]))
                    inxTot += inx


                # wide QRS (4)
                if (np.max(aux[:23]) > 0.0425 and not high_slope_flag) or \
                   (np.max(aux[10:25]) > 0.06) or \
                   (np.max(aux[:25]) > 0.029 and slur_flag and norm_coeff > 0.18) or \
                   np.max(aux[:25]) > 0.03 and QS_flag:
                    inx1 = np.where(np.max(aux[:20]) == aux[:20])[0][0]
                    log('MeasureQRSWidth J-Point.2.3.6 inx1: %s' % inx1)
                    aux_tmp = aux[inx1: inx1 + 20]
                    inx = np.where(np.abs(aux_tmp) <= aux[inx1]/4)[0]
                    if len(inx) > 0:
                        log('MeasureQRSWidth J-Point.2.3.6.1 inx: %s' % inx[0])
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

        inx = np.where(aux > -8e-3)[0]
        if not QS_flag and len(inx) == 0:
            log('MeasureQRSWidth QRS onset.0')
            low_ci_flag_2 = True  # discard measurement; no derivative >=0 to the left of the QRS

        if np.min(aux[:20]) <= -1.1e-2 and QS_flag and (qrs_peak_inx == p1 - inxs) and\
                ((beat_wave_lpf_norm[qrs_peak_inx] - beat_wave_lpf_norm[qrs_peak_inx - 2]) > -1.2e-2):#\
                # and (beat_wave_lpf_norm[qrs_peak_inx] - beat_wave_lpf_norm[jp]) > 0.25: # try removing the last condition
                                                                                        # example: file '30956-278c72c2527', K2500.
            # overrule QS-like morphology detection
            log('MeasureQRSWidth QRS onset.1')
            norm_coeff = np.abs(beat_avg[qrs_peak_inx])
            QS_flag = False

        if QS_flag and np.min(aux[:11]) > -0.038 and qrs_peak_inx == p1 - inxs:  # QS-like morphology
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
            aux2 = np.diff(aux)  # 2nd derivative
            inx1 = np.where(aux > 0)[0]  # derivative changes sign
            if len(inx1) == 0:
                inx1 = [np.Inf]

            thresh1 = 0.02 if norm_coeff < 0.45 else 0.01

            inx2 = np.where(np.abs(aux) <= thresh1)[0]
            if len(inx2) == 0:
                inx2 = [np.Inf]

            inx3 = np.where(aux2 < 5e-4)[0]
            if len(inx3) < 3 or not QS_flag:
                inx3 = [np.Inf]
            else:
                inx3 = inx3[inx3 > 8]

            inx = int(np.min([inx1[0], inx2[0], inx3[0]]))
            inxTot += inx
            aux = aux[inx:]  # QRS complex with no Q deflection

            # slurring QRS onset
            if np.min(aux[:10]) < -2.75e-2 and norm_coeff < .25:
                log('MeasureQRSWidth QRS onset.3.2')
                inx = np.where(aux[:10] == np.min(aux[:10]))[0][0]
                inx2 = np.where(np.abs(aux[inx:25]) < 3e-3)[0]
                if inx2.size > 0:
                    aux = aux[inx + inx2[0]:]
                    inxTot += inx + inx2[0]

            thresh1 = 5e-3 if norm_coeff > .66 else 0.0125

            # 2nd pass
            # look for Q deflection
            if np.max(aux[:12]) >= thresh1 and np.min(aux[1:10]) > -5e-2:
                log('MeasureQRSWidth QRS onset.3.3')
                aux_tmp = aux
                inx1 = np.where(aux[:12] == np.max(aux[:12]))[0][0]
                aux = aux[inx1:]
                aux2 = np.diff(aux)

                thresh1 = 6e-3 if norm_coeff < 0.32 else 3e-3

                inx2 = np.where(np.abs(aux) <= thresh1)[0]
                if len(inx2) == 0:
                    inx2 = [np.Inf]

                inx3 = np.where(aux < 0)[0]
                if len(inx3) == 0:
                    inx3 = [np.Inf]

                inx4 = np.where(aux2 > 1e-8)[0]
                if len(inx4) > 0 and (inx4[0] > 3 or np.max(aux2[:7]) < 5e-4) and not QS_flag:
                    log('MeasureQRSWidth QRS onset.3.3.1')
                    inx4 = inx4[inx4 >= 4][0] - 2
                else:
                    log('MeasureQRSWidth QRS onset.3.3.2')
                    inx4 = np.Inf

                inx5 = int(np.min([inx2[0], inx3[0], inx4]))
                inx = inx1 + inx5

                thresh1 = -3.5e-3 if norm_coeff > 0.18 else -0.015
                if (inx <= 18 and np.max(aux[inx5 + 1:inx5 + 1 + 5]) < 0.0125
                    and not (np.min(aux[inx5:inx5 + 25]) < thresh1 and inx >= 9 and np.max(aux[inx5:inx5 + 20]) < 0.013)) or \
                        (beat_avg[qrs_peak_inx] > 1.1 and np.min(aux[inx5:inx5 + 17]) > -2.1e-3) or \
                        (inx <= 19 and inx5 == inx4 and np.max(aux[10:20]) > 0.015) or \
                        (inx <= 19 and np.abs(inx5 - inx4) <= 1 and np.min(aux[inx5 + 4:inx5 + 20]) > -3.5e-3) or \
                        (QS_flag and (np.max(aux[:15]) > 6.7e-2 or aux[0] > 0.032)):

                    log('MeasureQRSWidth QRS onset.3.3.3')
                    inxTot += np.max([inx - 2, 0])
                    aux = aux[np.max([inx - 2, 0]):]
                else: # rollback
                    log('MeasureQRSWidth QRS onset.3.3.4')
                    aux = aux_tmp
            elif np.min(aux[1:12]) < -0.0174 and np.max(aux[5:15]) < -2e-3 and norm_coeff > 0.4 or\
                 np.min(aux[1:12]) < -0.03 and np.max(aux[5:15]) < 1e-4 and norm_coeff > 0.3:
                # slurring QRS
                log('MeasureQRSWidth QRS onset.3.4')
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
                    log('MeasureQRSWidth QRS onset.3.4.1')
                    aux = aux_tmp
                    inxTot = inxTotCpy

            # multiwave or QS-like morphologies
            if (aux[0] > 0.17 and np.min(aux[:15]) < -0.11) or \
                    (aux[0] > 0.17 and np.min(aux[:15]) < -0.05 and qrs_peak_inx == (p2 - inxs)) or \
                    (aux[0] > 0.05 and np.min(aux[:15]) < -0.079) or\
                    (aux[0] > 0.06 and np.min(aux[:15]) < -0.035) or\
                    (np.min(aux[:15]) < -0.2 and np.max(aux[:15]) > -0.12 and qrs_peak_inx == (p2 - inxs)) or \
                    (np.min(aux[:20]) < -0.15 and np.max(aux[:20]) < 0.04) or \
                    (np.min(aux[:21]) < -0.06 and np.max(aux[:20]) > 0.1 and norm_coeff < 0.25) or \
                    (np.min(aux[:20]) < -0.015 and np.max(aux[:20]) > 0.18 and norm_coeff > 0.8) or \
                    (QS_flag and qrs_peak_inx == p1 - inxs) or \
                    (QS_flag and np.min(aux[:20]) < -1.4e-2 and norm_coeff > 0.07) or \
                    (QS_flag and np.min(aux[:5]) < -0.05 and np.max(aux[:20]) > 7.5e-2) or \
                    beat_wave_lpf_norm[qrs_peak_inx - 1 - inxTot] < -0.95:
                log('MeasureQRSWidth QRS onset.3.5')
                inx0 = np.where(aux[1:15] == np.min(aux[1:15]))[0][0]
                inx1 = np.where(np.abs(aux[inx0:]) < 1e-2)[0][0]

                if inx0 + inx1 < 36:
                    log('MeasureQRSWidth QRS onset.3.5.1')
                    inxTot += inx0 + inx1 + 1
                    aux = aux[inx0 + inx1 + 1:]
                    if np.max(aux[:6]) > 7.5e-2:
                        log('MeasureQRSWidth QRS onset.3.5.1.1')
                        inx0 = np.where(aux[:6] == np.max(aux[:6]))[0][0]
                        inx1 = np.where(np.abs(aux[inx0:]) < 1e-2)[0][0]
                        inxTot += inx0 + inx1
                        aux = aux[inx0 + inx1:]
                    elif np.min(aux[:10]) < -0.065:
                        log('MeasureQRSWidth QRS onset.3.5.1.2')
                        inx0 = np.where(aux[:10] == np.min(aux[:10]))[0][0]
                        inx1 = np.where(np.abs(aux[inx0:]) < 1e-2)[0][0]
                        inxTot += inx0 + inx1
                        aux = aux[inx0 + inx1:]

            # Wide Q-wave
            if (np.max(aux[5:20]) > 6.6e-2 and np.min(aux[25:35]) > -2.5e-3) or \
                (QS_flag and qrs_peak_inx == p2 - inxs and np.max(aux[5:20]) > 6e-2 and np.min(aux[25:40]) > 0):
                log('MeasureQRSWidth QRS onset.3.6')
                inx0 = np.where(aux[5:20] == np.max(aux[5:20]))[0][0]
                inx1 = np.where(np.abs(aux[inx0 + 5:]) < 5e-3)[0][0]
                if inx1.size > 0:
                    log('MeasureQRSWidth QRS onset.3.6.1')
                    aux = aux[inx0 + inx1 + 5:]
                    inxTot += inx0 + inx1 + 5
                    if np.max(aux[5:15]) > 0.028 and np.min(aux[5:25]) > -2.5e-3 and not QS_flag:
                        log('MeasureQRSWidth QRS onset.3.6.1.1')
                        inx0 = np.where(aux[:15] == np.max(aux[:15]))[0][0]
                        inx1 = np.where(np.abs(aux[inx0:inx0 + 20]) <= 4e-3)[0][0]
                        inxTot += inx0 + inx1
                        aux = aux[inx0 + inx1:]

            # Wide QRS
            inx1 = np.where(np.min(aux[:10]) == aux[:10])[0][0]
            if norm_coeff < 0.5 and aux[inx1] < -0.035 and np.sum(aux[1:10] > 0) < 2:
                inx = np.where(aux[inx1:15] > 0)[0]
                log('MeasureQRSWidth QRS onset.3.7')
                if inx.size > 0:
                    log('MeasureQRSWidth QRS onset.3.7.1')
                    inxTot += inx1 + inx[0] - 2
                    aux = aux[inx1 + inx[0]:]

            if beat_wave_lpf_norm[qrs_peak_inx - 1 - inxTot] > 0.85 and \
                np.min(aux[:25]) < -8e-2:
                log('MeasureQRSWidth QRS onset.3.8')
                inx1 = np.where(aux[:25] == np.min(aux[:25]))[0][0]
                inx2 = np.where(np.abs(aux[inx1:]) < 1.5e-2)[0]
                if inx2.size > 0:
                    log('MeasureQRSWidth QRS onset.3.8.1')
                    inxTot += inx1 + inx2[0]

            # QS morphology refinement
            elif QS_flag and qrs_peak_inx == p2 - inxs and np.max(aux[:10]) > 2.5e-2:
                inx1 = np.where(aux[:10] == np.max(aux[:10]))[0][0]
                log('MeasureQRSWidth QRS onset.3.9')
                if np.min(aux[inx1: inx1 + 30]) > -3e-3:
                    inx2 = np.where(np.abs(aux[inx1:inx1 + 10]) < 2e-3)[0]
                    log('MeasureQRSWidth QRS onset.3.9.1')
                    if inx2.size > 0:
                        # aux = aux[inx1 + inx2[0]:]
                        inxTot += inx1 + inx2[0]
                        log('MeasureQRSWidth QRS onset.3.9.1.1')
            elif QS_flag and qrs_peak_inx == p2 - inxs and aux[0] > 6e-3:
                inx = np.where(np.abs(aux) < 2e-3)[0]
                log('MeasureQRSWidth QRS onset.3.10')
                if inx.size > 0:
                    # aux = aux[inx[0] + 2:]
                    inxTot += inx[0] + 2
                    log('MeasureQRSWidth QRS onset.3.10.1')

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
        if cond_det_4 and cond_det and cond_det_5 and not wavy_baseline_flag:
            log('MeasureQRSWidth P Wave.6')

            inx0 = np.where(peaks_properties['peak_heights'] == np.max(peaks_properties['peak_heights']))[0][0]
            p_inx = peaks[inx0]
            inxTot = p_inx

            cond2 = peaks_properties['prominences'][inx0] < 0.018 \
                    and polarity * beat_avg0[qrs_start_inx] > 1.12 * peaks_properties['peak_heights'][inx0]
            cond4 = (len(aux) - p_inx) <= 11
            if len(peaks) == 2:
                log('MeasureQRSWidth P Wave.6.1')
                cond6 = (np.max(peaks_properties['prominences']) / np.min(peaks_properties['prominences']) < 1.45)
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
                inx = np.where(aux2 >= 0.4 * aux2[0])[0]

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

                    # slurring baseline
                    bl_level = np.min(polarity * beat_avg0[p_inx:qrs_start_inx])
                    if polarity * beat_avg0[p_start_inx] < bl_level:
                        log('MeasureQRSWidth P Wave.6.3.3')
                        inx = np.where(polarity * beat_avg0[p_start_inx:] >= bl_level)[0][0]
                        p_start_inx += inx - 1


                    cond2_2 = (p_inx - p_start_inx) <= 7
                    cond2_3 = np.abs(beat_avg0[p_start_inx - search_shift - 1] -
                                     beat_avg0[p_inx - search_shift - 1]) < 0.009
                    cond2_5 = np.abs(beat_avg0[p_start_inx - search_shift - 1] -
                                     beat_avg0[p_inx - search_shift - 1]) < 0.027 \
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
        avg_beat_features['inxs'] = inxs  # index offset of the annotations w.r.t. the non-clipped average beat
                                          # (the function input average beat)

        return ([], avg_beat_features)

    except Exception as e:
        log('Exception: ', str(e))
        return ('QRS measurement error.', avg_beat_features)
