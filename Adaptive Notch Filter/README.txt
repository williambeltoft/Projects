GNSS LFM Jammer Mitigation & Signal Processing Pipeline
This repository contains a comprehensive suite of digital signal processing (DSP) tools and GNU Radio Embedded Python blocks designed to detect, estimate, and filter Linear Frequency Modulated (LFM) chirp jammers from GNSS (GPS L1 C/A) signals.

The project includes an end-to-end simulation environment, a highly accurate sub-bin peak detection algorithm, phase-continuous Local Oscillator (LO) generation, and real-time Adaptive Notch Filtering (ANF) utilizing an Overlap-Add (OLA) architecture.

Core Features
LFM Jammer Parameterization: Accurately estimates chirp rate, bandwidth, center frequency, and sweep period from raw IQ data using spectrogram analysis and linear regression.

Sub-Bin Peak Detection: Utilizes parabolic interpolation on spectrogram magnitude bins to resolve peak frequencies with sub-bin accuracy, preventing quantization errors.

Real-Time Drift Tracking: Actively measures and compensates for jammer timing drift and frequency offsets to maintain phase continuity over long acquisitions.

Phase-Continuous LO Generation: Features flyback blanking (Tukey windowing) to suppress parameterization timing errors at sweep resets.

Continuous Overlap-Add (OLA) Filtering: Implements a Kaiser-windowed FIR notch filter designed dynamically based on maximum allowed phase shift constraints.

GNU Radio Integration: Fully encapsulated signal processing logic packaged into standard GNU Radio asynchronous message-passing and sync blocks.

Project Structure
1. Signal Processing Library (JammerSignalProcessor)
The core utility class containing static methods for all heavy-lifting DSP operations. It requires no instantiation and handles:

peak_detection: Parabolic interpolation of spectrogram peaks.

parameterize_jammer: Segmenting sweeps and calculating physical jammer parameters.

calculate_filter_parameters & dc_firwin_filter: Dynamic FIR filter design.

LO_calculator: Downconversion and upconversion local oscillator synthesis.

refine_parameterization: Iterative refinement of jammer parameters.

2. GNSS & Jammer Simulation (GoldCodeGenerator)
A robust simulation class capable of generating realistic test vectors.

Generates GPS L1 C/A PRN sequences (e.g., PRN 27) with Doppler shift simulation capabilities.

Generates target LFM chirp signals with optional, configurable Gaussian frequency jitter to simulate hardware imperfections in real jammers.

3. GNU Radio Embedded Blocks
The repository provides out-of-tree (OOT) equivalent embedded blocks for real-time SDR processing.

EstimatorBlock: Buffers a specific chunk of incoming IQ data, performs the initial spectrogram physics estimation, and broadcasts the parameters downstream via a Polymorphic Type (PMT) dictionary message (jammer_params_out).

AdaptiveNotchFilterBlock: Subscribes to the PMT messages, initializes the FIR filter, tracks active phase drift, and executes the continuous OLA downconvert-filter-upconvert loop.

NormalizedPeakCorrelator: An analysis block that computes FFT-based cross-correlation, returning a normalized score (0 to 1) to evaluate the success of the mitigation against a clean reference signal.

Dependencies
To run the simulation and utilize the DSP classes, the following Python libraries are required:

numpy

scipy

matplotlib

pandas

seaborn

tqdm

gnuradio (Specifically for the Embedded Python Blocks)

Getting Started
Run the Simulation: The included simulation script sets up a 2.56 MHz environment, generates 10 seconds of GNSS PRN 27 data combined with an 80 dB LFM jammer, and processes it sequentially. Run this script to visualize the parameterization accuracy and the recovered signal's correlation score.

GNU Radio Integration: To use the blocks in GNU Radio Companion (GRC), instantiate Embedded Python Blocks, copy the respective class code (EstimatorBlock, AdaptiveNotchFilterBlock, NormalizedPeakCorrelator) into the block editors, and connect the asynchronous message ports (jammer_params_out to jammer_params_in).

Credits
William Vauvert Beltoft (DTU, Denmark) - Core architecture, DSP logic, peak detection, drift tracking, and GNU Radio block implementation.

Benjamin Aywaz (DTU, Denmark) - Looped OLA code

Lasse Lehmann (Postdoc at DTU Space, Denmark) - Original author of the GoldCodeGenerator logic.
