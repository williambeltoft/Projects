"""
Author: William Vauvert Beltoft, student at DTU, Denmark
Exceptions where AI or external help was used are marked with '''Made with: ...''' comments. The overall code structure and all core logic were designed and implemented by the author, with AI assistance primarily for code organization and minor implementation details.
Class structured by: Gemini AI
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy import stats
import scipy.signal
from scipy.signal import firwin2


@dataclass
class JammerParams:
    chirp_rate:      float          # Hz/s — slope of f(t)
    bandwidth:       float          # Hz — f_stop - f_start (per sweep)
    f_center:        float          # Hz — midpoint of sweep band
    f_start:         float          # Hz — per-sweep start (median)
    f_stop:          float          # Hz — per-sweep stop (median)
    sweep_period:    float          # s  — repetition interval
    duty_cycle:      float          # 0-1, active/period
    sweep_type:      Literal["up", "down", "triangle", "unknown"]
    linearity_score: float          # R² of linear fit, 0–1
    n_sweeps:        int            # number of full sweeps detected
    residual_rms:    float          # Hz — RMS fit residual


class JammerSignalProcessor:
    """
    A utility class containing all signal processing, peak detection, and jammer analysis tools.
    All methods are static, allowing them to be called without instantiating the class.
    """    
    '''
    Author: William Vauvert Beltoft, student at DTU, Denmark
    '''
    @staticmethod
    def calculate_filter_parameters(chirp_rate, f_samplerate, bandwidth, notch_width_percentage, transition_width_multiplier, max_phase):
        
        """
        Calculates the required filter length (number of taps) based on the 
        maximum allowed phase shift.

        The derivation follows the constraint that the maximum reachable phase 
        shift $\Phi_M$ occurs at the highest absolute index $m = M$. 
        By setting a threshold $\Phi_{max}$, we establish the relation:

        $$\pi \alpha M^2 \leq \Phi_{max}$$

        By substituting $M = \frac{N_{taps}}{2}$ (assuming $N \approx N-1$) 
        and isolating $N_{taps}$, we derive the filter length as a function 
        of the maximum allowed phase shift:

        $$N_{taps} = 2 \cdot \sqrt{\frac{\Phi_{max}}{\pi \alpha}}$$

        Args:
            phi_max (float): Maximum allowed phase shift $\Phi_{max}$ in radians.
            alpha (float): Phase acceleration constant $\alpha$.

        Returns:
            int: The required number of taps $N_{taps}$.
        """
        
        chirp_rate_discrete = np.abs(chirp_rate) / f_samplerate**2 
        M_max = np.sqrt(max_phase/(np.pi * chirp_rate_discrete))
        N_taps = int(np.ceil(2*M_max)) + 1
        
        notch_width = notch_width_percentage * bandwidth/2
        transition_width_Hz = notch_width * transition_width_multiplier

        if N_taps % 2 == 0: 
            N_taps += 1

        return N_taps, notch_width, transition_width_Hz

    
    @staticmethod
    def LO_calculator(f_start, chirp_rate, sweep_period, t_global, taper_sec, t_sync=0.0):
        """
        Create phase-continuous local oscillators with Flyback Blanking 
        to mitigate parameterization timing errors at sweep resets.
        """
        # Align the global time to the parameterized grid
        t_aligned = t_global - t_sync
        
        # Determine exactly where we are in the current sweep cycle
        cycle_number = np.floor(t_aligned / sweep_period).astype(int) 
        t_in_sweep = t_aligned - (cycle_number * sweep_period) 
        
        # Calculate continuous analytical phase
        phase_in_sweep = f_start * t_in_sweep + 0.5 * chirp_rate * (t_in_sweep**2) 
        phase_per_cycle = f_start * sweep_period + 0.5 * chirp_rate * (sweep_period**2) 
        total_phase = phase_in_sweep + (cycle_number * phase_per_cycle)
        
        phase_down = 2 * np.pi * total_phase 
        LO_down = np.exp(-1j * phase_down) 
        
        # The fix for the spike at the chirp reset was to simply discard the data entirely. Thats why we 
        # introduce a blanking envelope that smoothly tapers the LO to zero at the edges of each sweep. We
        # willingly throw away 200 microseconds of GNSS data to save the rest of the signal.
        
        envelope = np.ones_like(t_in_sweep)
        
        # Smoothly taper down the rising edge of the reset
        mask_rise = t_in_sweep < taper_sec
        envelope[mask_rise] = 0.5 * (1 - np.cos(np.pi * t_in_sweep[mask_rise] / taper_sec))
        
        # Smoothly taper down the falling edge of the reset
        mask_fall = t_in_sweep > (sweep_period - taper_sec)
        envelope[mask_fall] = 0.5 * (1 - np.cos(np.pi * (sweep_period - t_in_sweep[mask_fall]) / taper_sec))
        
        # Apply the blanking envelope
        LO_down = LO_down * envelope
        
        # Upconversion restores phase but maintains the boundary blanking
        LO_up = np.conj(LO_down) 
        
        return LO_down, LO_up


    @staticmethod
    def peak_detection(signal_chunk, chunk_size, chunk_overlap, f_samplerate):
        """
        This peak detection method is a step up from the simple argmax approach. 
        We find the neighboring bins around the argmax peak, and perform an interpolation.
        Whereas argmax only finds the single bin with the highest magnitude, this method uses the shape of the spectrum around the peak to estimate where the true peak lies between bins.
        """
        # Signal Spectogram splits the signal into len(Sxx[:][0]) blocks
        f_axis, t_axis , Sxx = scipy.signal.spectrogram(signal_chunk, 
                                                        f_samplerate, 
                                                        return_onesided=False, 
                                                        nperseg=chunk_size, 
                                                        noverlap=int(chunk_overlap), 
                                                        mode='magnitude', 
                                                        detrend=False)

        # Find the peak frequency idx for every block, argmax(axis=0) looks down each column to find the row index of the max value
        max_indices = np.argmax(Sxx, axis=0)
        detected_frequencies = np.zeros(len(max_indices)) # We initialize the array to store the detected frequencies for each block
        bin_width = f_axis[1] - f_axis[0] # Calculate the width of each frequency bin

        for i, k in enumerate(max_indices):
            if k == 0 or k == len(f_axis) - 1:
                detected_frequencies[i] = f_axis[k] # If the peak is at the edge, we can't do interpolation, so we just take the bin frequency
            else:                           # We pick three points: the peak bin and its immediate neighbors to perform parabolic interpolation.
                left   = Sxx[k - 1, i]      # The magnitude of the bin to the left of the peak
                center = Sxx[k, i]          # The magnitude of the peak bin
                right  = Sxx[k + 1, i]      # The magnitude of the bin to the right of the peak
                
                '''
                The parabolic interpolation is made with the logic, that if the true peak is between bins, the neighboring bins will be asymmetrically lower. 
                The formula calculates a correction factor 'p' that estimates how far the true peak is from the center of the bin, based on the relative magnitudes of the left and right neighbors compared to the center. 
                This allows us to adjust our detected frequency to be more accurate than just taking the center bin frequency.
                This can be described with the following math formulas:
                correction = 0.5 * (left - right) / (left - 2*center + right)
                '''
                denominator = left - 2 * center + right
                if denominator == 0:
                    correction = 0.0
                else:
                    correction = 0.5 * (left - right) / denominator # Parabolic interpolation formula
                
                detected_frequencies[i] = f_axis[k] + (correction * bin_width) # Adjust the detected frequency by the interpolated correction factor, scaled by the bin width to convert from bin index to frequency offset

        # Peak freqs in Hz
        detected_frequencies = f_axis[max_indices]

        return detected_frequencies, t_axis    


    @staticmethod
    def dc_firwin_filter(N_taps, f_samplerate, width_in_Hz, attenuation_dB, transition_width_Hz):
        # Sørg for ulige taps for symmetri
        if N_taps % 2 == 0:
            N_taps += 1
            
        # Beregn beta til Kaiser-vinduet (samme logik som før)
        if attenuation_dB > 50:
            beta = 0.1102 * (attenuation_dB - 8.7)
        else:
            beta = 0.5842 * (attenuation_dB - 21)**0.4 + 0.07886 * (attenuation_dB - 21)

        # Definer de kritiske frekvenspunkter (normaliseret til Nyquist = 1.0)
        nyq = f_samplerate / 2
        
        # 2. Beregn punkterne med sikkerhedsmarginer
        # Vi bruger np.clip for at sikre, at vi aldrig rammer præcis 0 eller Nyquist
        # og sikrer at de er strengt stigende
        f1 = 0.0
        f2 = np.clip(width_in_Hz / 2, 1e-6, nyq - 2e-6)
        f3 = np.clip(f2 + transition_width_Hz, f2 + 1e-6, nyq - 1e-6)
        f4 = nyq

        # Frekvens-vektor: Går fra 0 (DC) til 1.0 (Nyquist)
        # Vi definerer de præcise knækpunkter i vores trapez
        freqs = [f1, f2, f3, f4]
        
        # Gain-vektor: Svarer til frekvenserne ovenfor
        gains = [0, 0, 1, 1]
        
        # firwin2 tager frekvenser, gains og vinduet direkte
        h = firwin2(
            numtaps =   N_taps,      # Antal filter-taps (brug et ulige tal for Type 1)
            freq =      freqs,    # Liste med frekvenser (Hz eller normaliseret 0-1)
            gain =      gains,    # Liste med gain ved hver frekvens (0.0 til 1.0)
            window =    ('kaiser', beta),    # Valgfrit: Vinduestype (f.eks. 'hamming' eller ('kaiser', beta))
            fs =        f_samplerate      # Valgfrit: Samplingsrate (hvis frekvenser er i Hz)
        )
        
        return h

    @staticmethod
    def analyze_correlation(reference_signal, dirty_signal):
        """
        Analyze the correlation between two signals using cross-correlation.
        
        Parameters
        ----------
        reference_signal : ndarray
            First input signal (Which is the clean/reference version)
        dirty_signal : ndarray
            Second input signal (Which is the dirty version of the first)
        
        Returns
        -------
        correlation : ndarray
            Cross-correlation values between reference_signal and dirty_signal
        """
        correlation = scipy.signal.correlate(
            reference_signal, 
            dirty_signal, 
            mode='full', 
            method='fft'
        )
        normalization_factor = np.sqrt(np.sum(np.abs(reference_signal)**2) * np.sum(np.abs(dirty_signal)**2))
        correlation = correlation / normalization_factor  # Normalize the correlation

        max_correlation = np.max(np.abs(correlation))

        return max_correlation


    @staticmethod
    def parameterize_jammer(detected_frequencies, t_axis, min_sweep_points=10):
        """
        Refactored parameterization that derives the TRUE physical period 
        by measuring center-to-center timing, making it immune to edge-fading.
        """
        # 1. Segment the signal into individual sweeps (Unchanged)
        df = np.diff(detected_frequencies)
        discontinuity_threshold = 5.0 * np.median(np.abs(df))
        break_indices = np.where(np.abs(df) > discontinuity_threshold)[0] + 1
        edges = np.concatenate([[0], break_indices, [len(detected_frequencies)]])
        slices = [slice(int(edges[i]), int(edges[i+1])) for i in range(len(edges)-1)]
        valid_segs = [s for s in slices if (s.stop - s.start) >= min_sweep_points]

        # 2. Fit each segment and find the MIDPOINT frequency and time
        chirp_rates, t_mids, f_mids = [], [], []
        r2_scores = []

        for sl in valid_segs:
            t_seg, f_seg = t_axis[sl], detected_frequencies[sl]
            slope, intercept, r, _, _ = stats.linregress(t_seg, f_seg)
            
            # Find the midpoint of this specific segment
            t_mid = (t_seg[0] + t_seg[-1]) / 2.0
            f_mid_fit = intercept + slope * t_mid
            
            chirp_rates.append(slope)
            t_mids.append(t_mid)
            f_mids.append(f_mid_fit)
            r2_scores.append(r**2)

        # 3. DERIVE TRUE PERIOD (Center-to-Center)
        # This is the key: The time between centers is the TRUE sweep period.
        if len(t_mids) > 1:
            period_med = float(np.median(np.diff(t_mids)))
        else:
            # Fallback for single-sweep chunks (less accurate)
            period_med = (np.max(detected_frequencies) - np.min(detected_frequencies)) / np.abs(np.median(chirp_rates))

        chirp_rate_med = float(np.median(chirp_rates))

        # 4. EXTRAPOLATE FULL PHYSICAL BANDWIDTH
        # We don't care where the red dots stop. We project the line out 
        # to fill the entire Period.
        full_bandwidth = abs(chirp_rate_med * period_med)
        
        # Calculate the f_start/f_stop relative to the midpoints
        f_start_extrapolated = float(np.median(f_mids)) - (chirp_rate_med * period_med / 2.0)
        f_stop_extrapolated = f_start_extrapolated + (chirp_rate_med * period_med)

        return JammerParams(
            chirp_rate      = chirp_rate_med,
            bandwidth       = full_bandwidth,
            f_center        = float(np.median(f_mids)),
            f_start         = f_start_extrapolated, # The true physical floor
            f_stop          = f_stop_extrapolated,  # The true physical ceiling
            sweep_period    = period_med,           # The true physical cycle
            duty_cycle      = 1.0, 
            sweep_type      = "up" if chirp_rate_med > 0 else "down",
            linearity_score = float(np.median(r2_scores)),
            n_sweeps        = len(valid_segs),
            residual_rms    = 0.0 # simplified
        )
    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
 
    @staticmethod
    def _indices_to_slices(
        break_indices: np.ndarray, total_len: int
    ) -> list[slice]:
        '''
        Made by Claude AI
        '''
        edges = np.concatenate([[0], break_indices, [total_len]])
        return [slice(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]
    
    
    @staticmethod
    def mock_lo_calculator(f_start, chirp_rate, sweep_period, t_vec):
        '''
        This is a simplified version of the LO calculator that generates a continuous-phase LO without any blanking or tapering.
        Kind of self-explanatory.
        '''
        t_rel = t_vec % sweep_period
        phase = 2 * np.pi * (f_start * t_rel + 0.5 * chirp_rate * t_rel**2)
        cycle_idx = np.floor(t_vec / sweep_period)
        phase_acc = cycle_idx * 2 * np.pi * (f_start * sweep_period + 0.5 * chirp_rate * sweep_period**2)
        return np.exp(-1j * (phase + phase_acc))
    
    @staticmethod
    def refine_parameterization(acq_chunk, coarse_params, f_samplerate, t_sync, start_idx, taper_sec):
        '''
        The original parameterization was a single guess, this simply iteratively refines it.
        We calculate the LO_down and measure how much it is off. Then we use that error feedback to refine the parameters.
        OBS! This uses the peak_detection() function with specific values for chunk_size and chunk_overlap, which are not necessarily optimal for all signals. 
        '''
        # 1. Test Drive: Downconvert using COARSE parameters
        t_global = (start_idx + np.arange(len(acq_chunk))) / f_samplerate
        
        # Use actual LO calculator to ensure flyback blanking and t_sync are respected
        LO_down_test, _ = JammerSignalProcessor.LO_calculator(
            coarse_params.f_start, 
            coarse_params.chirp_rate, 
            coarse_params.sweep_period, 
            t_global,
            taper_sec,
            t_sync
        )
        
        residual_signal = acq_chunk * LO_down_test
        
        
        chunk_size = 1024
        chunk_overlap = int(0.8 * chunk_size)

        # 2. Find the peaks in the residual signal's spectrogram to see how far off we are.
        f_res, t_res = JammerSignalProcessor.peak_detection(residual_signal, 
                                                            chunk_size, 
                                                            chunk_overlap,
                                                            f_samplerate    = f_samplerate
        )
        
        # Unwrap frequencies to avoid wrap-around at Fs/2 if the drift is large. 
        # This ensures that the linear regression captures the true slope and offset, rather than being confused by jumps in the detected frequency.
        f_res_unwrapped = np.unwrap(f_res * 2 * np.pi / f_samplerate) * f_samplerate / (2 * np.pi)
        
        # 3. Global Linear Regression on the Residual
        delta_k, delta_f_start, r_value, _, _ = stats.linregress(t_res, f_res_unwrapped)


        # 4. Measure Timing Drift (Period Error)
        # By finding the phase discontinuities (the "jumps" in the unwrapped residual),
        # we can calculate the exact time between sweeps over the full 0.5s baseline.
        df = np.abs(np.diff(f_res))
        jump_indices = np.where(df > np.std(df) * 3)[0] # Find the sweep reset boundaries
        
        if len(jump_indices) > 1:
            t_jumps = t_res[jump_indices]
            # Macroscopic averaging: (Last Jump - First Jump) / Number of Cycles
            # This mathematically eliminates the spectrogram's hop-size quantization error.
            true_period = (t_jumps[-1] - t_jumps[0]) / (len(t_jumps) - 1)
        else:
            true_period = coarse_params.sweep_period # Fallback if no jumps found
            
        print(f"Residual Error Detected: \n- Chirp Drift = {delta_k/1e3:.2f} kHz/s \n- Freq Offset = {delta_f_start/1e3:.2f} kHz \n- Period Correction = {(true_period - coarse_params.sweep_period)*1e6:.2f} µs")
  
        
        # 5. Apply Independent Corrections
        coarse_params.chirp_rate += delta_k
        coarse_params.f_start += delta_f_start
        coarse_params.sweep_period = true_period 
        
        # Bandwidth is derived dynamically from the newly corrected independent variables
        coarse_params.bandwidth = abs(coarse_params.chirp_rate * coarse_params.sweep_period)
        coarse_params.f_stop = coarse_params.f_start + coarse_params.bandwidth
        
        refined_params = coarse_params
        
        return refined_params

    @staticmethod
    def estimate_jammer_physics(acquisition_chunk, f_samplerate, start_idx, taper_sec, spectrogram_nperseg, spectrogram_overlap):
        '''
        This function is the "Master Function" of sorts. It calls the other functions and performs the full parameter estimation and refinement process. 
        It is designed to be called in a loop over acquisition chunks, where each chunk is processed to extract and refine jammer parameters.
        '''
        detected_frequencies, t_axis_relative = JammerSignalProcessor.peak_detection(
                                                                                    acquisition_chunk,
                                                                                    spectrogram_nperseg,
                                                                                    spectrogram_overlap, 
                                                                                    f_samplerate
                                                                                    )

        coarse_params = JammerSignalProcessor.parameterize_jammer(detected_frequencies, t_axis_relative)

        # 1. Test Drive: Downconvert using COARSE parameters
        t_global = (start_idx + np.arange(len(acquisition_chunk))) / f_samplerate
        
        f_detect = detected_frequencies[0]
        t_detect_global = (start_idx / f_samplerate) + t_axis_relative[0]
        t_sync = t_detect_global - ((f_detect - coarse_params.f_start) / coarse_params.chirp_rate)

        # Use actual LO calculator to ensure flyback blanking and t_sync are respected
        LO_down_test, _ = JammerSignalProcessor.LO_calculator(
            coarse_params.f_start, 
            coarse_params.chirp_rate, 
            coarse_params.sweep_period, 
            t_global,
            taper_sec,
            t_sync
        )
        
        residual_signal = acquisition_chunk * LO_down_test
        
        
        chunk_size = 1024
        chunk_overlap = int(0.8 * chunk_size)

        # 2. Find the peaks in the residual signal's spectrogram to see how far off we are.
        f_res, t_res = JammerSignalProcessor.peak_detection(residual_signal, 
                                                            chunk_size, 
                                                            chunk_overlap,
                                                            f_samplerate    = f_samplerate
        )
        
        # Unwrap frequencies to avoid wrap-around at Fs/2 if the drift is large. 
        # This ensures that the linear regression captures the true slope and offset, rather than being confused by jumps in the detected frequency.
        f_res_unwrapped = np.unwrap(f_res * 2 * np.pi / f_samplerate) * f_samplerate / (2 * np.pi)
        
        # 3. Global Linear Regression on the Residual
        delta_k, delta_f_start, r_value, _, _ = stats.linregress(t_res, f_res_unwrapped)
        
        print(f"Residual Error Detected: Chirp Drift = {delta_k/1e3:.2f} kHz/s\nFreq Offset = {delta_f_start/1e3:.2f} kHz\nR² of Fit = {r_value**2:.4f}")
        
        # 4. Apply Corrections directly to the JammerParams object attributes
        coarse_params.chirp_rate += delta_k
        coarse_params.f_start += delta_f_start
        coarse_params.sweep_period = coarse_params.bandwidth / abs(coarse_params.chirp_rate)
        coarse_params.f_stop = coarse_params.f_start + coarse_params.bandwidth
        coarse_params.main_lobe_width = 1.0 / coarse_params.bandwidth
        coarse_params.t_sync = t_sync

        refined_params = coarse_params
        
        return refined_params # Which is now refined params. A bit confusing I know.




"""
GPS L1 C/A Gold code generator
Made by: Lasse Lehmann, Postdoc at DTU Space, Denmark
"""
class GoldCodeGenerator:
    def __init__(self, PRN=1, samp_rate=1.023e6, vectorLength=5120, dopplerSpan=5e3, dopplerStep=250):

        self.PRN = int(PRN)
        self.samp_rate = float(samp_rate)
        self.chips = 1.023e6
        self.samples_per_code = int(round(1023 * self.samp_rate/self.chips))
        self.g2s = [5, 6, 7, 8, 17, 18, 139, 140, 141, 251, 252, 254, 
                    255, 256, 257, 258, 469, 470, 471, 472, 473, 474, 
                    509, 512, 513, 514, 515, 516, 859, 860,861, 862, 
                    145, 175, 52, 21, 237, 235, 886, 657, 634, 762, 
                    355, 1012, 176, 603, 130, 359, 595, 68, 386]
        self.vectorLength = vectorLength
        # Precompute the full 1023-chip CA code once
        # self.code = self._generate_code(self.PRN).astype(np.complex64)
        self.idx_chip = 0   # current position in the sequence (chip counter)
        self.idx_doppler = 0

        self.Doppler_vec = np.arange(-dopplerSpan, dopplerSpan, dopplerStep)

        # FIX: generate Gold code once
        self.code = self._generate_code(self.PRN).astype(np.complex64)
                    

    def process(self): # [GR:] process(self, input_items, output_items):
        out = np.zeros((self.vectorLength), dtype=np.complex64)  # [GR:] out = output_items[0]
        n = len(out)  # number of vectors to produce

        # Hardcode that there is no apparent Doppler-shift
        frequency_shift = 0 # self.Doppler_vec[self.idx_doppler] # [int(self.idx_doppler*(1/self.vectorLength))]
        frequency_phasor = self._generate_frequency_phasor(frequency_shift)

        # For this vector, compute chip index offset
        chip_index = int((self.idx_chip * self.chips) / self.samp_rate)
        resampleFactor = 1.023e6/self.samp_rate

        # FIX 2: output a vector, not a scalar
        for k in range(self.vectorLength):
            out[k] = self.code[int(np.floor((k*resampleFactor) +chip_index)) % 1023] #* frequency_phasor

        # Create BPSK phasor, modulated in accordance with Doppler-phasor
        out = np.exp(0.5j * np.pi*out)*frequency_phasor

        # Advance sample counter by *vector length* worth of samples
        self.idx_chip += self.vectorLength
        self.idx_doppler += 1

        # FIX 3: proper wrap
        if self.idx_chip >= self.samples_per_code:
            self.idx_chip = 0
        if self.idx_doppler >= len(self.Doppler_vec):
            self.idx_doppler = 0

        return out

    def _generate_frequency_phasor(self, frequency_shift):
        index_vector = np.arange(0, self.vectorLength)
        sample_duration = 1/self.samp_rate

        time_vector  = index_vector * sample_duration

        frequency_phasor = np.exp(2j * np.pi * frequency_shift * time_vector)
        return frequency_phasor


    def _generate_code(self, prn):

        g2shift = self.g2s[prn-1]

        # Generate G1 code.
        # Initialize g1 output to speed up the function.
        g1 = [0] * 1023
        # Load shift register.
        reg = [-1] * 10
        # Generate all G1 signal chips based on the G1 feedback polynomial.
        for i in range(1023):
            g1[i] = reg[9]
            saveBit = reg[2] * reg[9]
            reg[1:10] = reg[0:9]
            reg[0] = saveBit

        # Generate G2 code.
        # Initialize g2 output to speed up the function.
        g2 = [0] * 1023
        # Load shift register.
        reg = [-1] * 10
        # Generate all G2 signal chips based on the G2 feedback polynomial.
        for i in range(1023):
            g2[i] = reg[9]
            saveBit = reg[1] * reg[2] * reg[5] * reg[7] * reg[8] * reg[9]
            reg[1:10] = reg[0:9]
            reg[0] = saveBit

        # Shift
        g2 = g2[1023-g2shift:1023] + g2[0:1023-g2shift]

        # Gold code
        return np.array([-1 * g1[i] * g2[i] for i in range(1023)])


    @staticmethod
    def create_linear_FM_signal(freqs, t_vec, f_samplerate, amplitude, chirp_rate = None, jitter_std = 0.0, taper_ratio = 0.0):
        """
        Generate a linear frequency modulated (chirp) signal with explicit chirp rate.
        
        Parameters
        ----------
        freqs : array of floats
            Frequencies defining the chirp sweep [Hz] - [start_freq, end_freq]
        t_vec : ndarray
            Time vector [seconds]
        f_samplerate : float
            Sampling rate [Hz]
        amplitude : float
            Amplitude of the chirp signal (default: 1.0)
        chirp_rate : float
            Rate of frequency change [Hz/s]. If None, calculates from freqs and t_vec length.
        jitter_std : float
            Standard deviation of Gaussian frequency noise to add [Hz] (default: 0.0, no noise)
        taper_ratio : float
            Ratio of the sweep duration used for cosine tapering at the edges (default: 0.1)
        Returns
        -------
        signal : ndarray
            Time-domain linear FM signal with continuous phase across sweep resets
        """
        # If chirp_rate not specified, calculate it from frequency sweep
        if chirp_rate is None:
            sweep_duration = (len(t_vec) - 1) / f_samplerate
            chirp_rate = (freqs[-1] - freqs[0]) / sweep_duration
        
        # Duration of one complete sweep
        sweep_duration = (freqs[-1] - freqs[0]) / chirp_rate
        
        # Determine which cycle each sample is in and time within that cycle
        cycle_number = np.floor(t_vec / sweep_duration).astype(int)
        t_in_sweep = t_vec - (cycle_number * sweep_duration)
        
        # Phase within each sweep
        phase_in_sweep = 2 * np.pi * (freqs[0] * t_in_sweep + 0.5 * chirp_rate * (t_in_sweep**2))
        
        # Phase accumulated from previous complete cycles
        phase_per_cycle = 2 * np.pi * (freqs[0] * sweep_duration + 0.5 * chirp_rate * (sweep_duration**2))
        phase_accumulated = cycle_number * phase_per_cycle
        
        # Total continuous phase
        phase = phase_in_sweep + phase_accumulated
        
        # --- ADD GAUSSIAN FREQUENCY NOISE ---
      # --- ADD GAUSSIAN FREQUENCY NOISE (CORRELATED / REALISTIC) ---
        if jitter_std > 0:
            # 1. Generate White Frequency Noise
            # This jumps wildly every sample (Bad for filtering)
            white_noise = np.random.normal(loc=0.0, scale=jitter_std, size=len(t_vec))
            
            # 2. THE FIX: Smooth the noise
            # We use a Gaussian filter to make the frequency drift "slowly" 
            # instead of instantly. Sigma=100 means the frequency changes 
            # typically over 100 samples, keeping the bandwidth narrow.
            from scipy.ndimage import gaussian_filter1d
            
            # Adjust sigma to control "speed" of drift. 
            # Higher = Slower drift = Easier to filter.
            freq_noise_Hz = gaussian_filter1d(white_noise, sigma=100) 
            
            # Optional: Re-normalize to maintain the requested jitter_std amplitude
            # (Filtering reduces variance, so we boost it back up)
            current_std = np.std(freq_noise_Hz)
            if current_std > 0:
                freq_noise_Hz = freq_noise_Hz * (jitter_std / current_std)

            # 3. Integrate to get Phase Jitter
            dt = 1.0 / f_samplerate
            phase_jitter = 2 * np.pi * np.cumsum(freq_noise_Hz) * dt
            
            # 4. Add to main phase
            phase += phase_jitter
        
        # Normalize time within sweep to [0, 1]
        norm_t = t_in_sweep / sweep_duration
        envelope = np.ones_like(norm_t)
        
        if taper_ratio > 0:
            # Define the width of the edge tapers (split ratio between start and end)
            edge_width = taper_ratio / 2.0
            
            # 1. Rising Edge (Start of sweep)
            mask_rise = norm_t < edge_width
            # Cosine ramp from 0 to 1
            envelope[mask_rise] = 0.5 * (1 - np.cos(np.pi * norm_t[mask_rise] / edge_width))
            
            # 2. Falling Edge (End of sweep)
            mask_fall = norm_t > (1 - edge_width)
            # Cosine ramp from 1 to 0
            envelope[mask_fall] = 0.5 * (1 - np.cos(np.pi * (1 - norm_t[mask_fall]) / edge_width))

            signal = amplitude * envelope * np.exp(1j * phase)
        
        # Create signal
        else:     
            signal = amplitude * np.exp(1j * phase)

        return signal

        

