import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import filtfilt, butter, resample, lfilter

# Load the audio files
morgan_freeman_file = 'morgan_freeman.wav'
swahili_reporter_file = 'swahili_reporter.wav'

morgan_freeman_rate, morgan_freeman_audio = wavfile.read(morgan_freeman_file)
swahili_reporter_rate, swahili_reporter_audio = wavfile.read(swahili_reporter_file)

# Analyze the Morgan Freeman audio
morgan_freeman_features = extract_voice_features(morgan_freeman_audio, morgan_freeman_rate)

# Apply RAGs to the Swahili reporter audio
rags_audio = apply_rags(swahili_reporter_audio, swahili_reporter_rate, morgan_freeman_features)

# Save the resulting audio
wavfile.write('morgan_freeman_swahili.wav', swahili_reporter_rate, rags_audio.astype(np.int16))

def extract_voice_features(audio, sample_rate):
    """
    Extract relevant voice features from the audio, such as pitch, formants, and spectral characteristics.
    """
    pitch = get_pitch(audio, sample_rate)
    formants = get_formants(audio, sample_rate)
    spectrum = get_spectrum(audio, sample_rate)
    
    return {
        'pitch': pitch,
        'formants': formants,
        'spectrum': spectrum
    }

def apply_rags(audio, sample_rate, voice_features):
    """
    Apply the RAGs process to the input audio, using the provided voice features.
    """
    pitch_shifted_audio = shift_pitch(audio, sample_rate, voice_features['pitch'])
    formant_adjusted_audio = adjust_formants(pitch_shifted_audio, sample_rate, voice_features['formants'])
    rags_audio = apply_spectral_transfer(formant_adjusted_audio, sample_rate, voice_features['spectrum'])
    
    return rags_audio

# Helper functions for the feature extraction and RAGs application
def get_pitch(audio, sample_rate):
    # Autocorrelation method for pitch extraction
    
    # Define parameters
    frame_length = 0.02  # 20 milliseconds
    frame_stride = 0.01  # 10 milliseconds
    frame_length_samples = int(frame_length * sample_rate)
    frame_stride_samples = int(frame_stride * sample_rate)
    
    # Calculate autocorrelation for each frame
    autocorr_values = []
    for i in range(0, len(audio) - frame_length_samples, frame_stride_samples):
        frame = audio[i:i+frame_length_samples]
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr_values.append(autocorr)
    
    # Find the lag corresponding to the maximum autocorrelation within a suitable range
    pitch_values = []
    for autocorr in autocorr_values:
        # Limit the search range for pitch based on human voice range (75 Hz to 500 Hz)
        min_lag = int(sample_rate / 500)
        max_lag = int(sample_rate / 75)
        lag_range = np.arange(min_lag, max_lag + 1)
        autocorr_segment = autocorr[len(autocorr)//2:]  # Use only positive lags
        pitch_lag = lag_range[np.argmax(autocorr_segment[min_lag:max_lag+1])]
        pitch_values.append(sample_rate / pitch_lag)
    
    return np.array(pitch_values)

def get_formants(audio, sample_rate):
    # Placeholder implementation for formant extraction
    # Replace this with your own formant extraction algorithm
    return np.array([])

def get_spectrum(audio, sample_rate):
    # Compute the magnitude spectrum using Short-Time Fourier Transform (STFT)
    return np.abs(librosa.stft(audio))

def shift_pitch(audio, sample_rate, pitch_shifts):
    # Convert pitch shifts to semitones
    semitone_shifts = pitch_shifts / 100  # Assuming pitch_shifts are in cents
    
    # Calculate the factor by which to resample the audio
    resample_factor = 2 ** (semitone_shifts / 12)
    
    # Resample the audio to shift its pitch
    shifted_audio = resample(audio, int(len(audio) * resample_factor))
    
    return shifted_audio

def adjust_formants(audio, sample_rate, formant_adjustments):
    # Convert formant adjustments to frequency shifts
    frequency_shifts = formant_adjustments * 100  # Adjustments are assumed to be in Hz, scaling for better effect
    
    # Define filter parameters for formant adjustment
    order = 8  # Filter order
    nyquist = 0.5 * sample_rate
    low = 300   # Lower bound for formant frequencies (Hz)
    high = 5000 # Upper bound for formant frequencies (Hz)
    
    # Create bandpass filters for each formant frequency range
    b, a = butter(order, [low/nyquist, high/nyquist], btype='band')
    
    # Apply formant adjustment to each frequency shift
    adjusted_audio = np.copy(audio)
    for shift in frequency_shifts:
        # Apply frequency shift using a bandpass filter
        adjusted_audio = filtfilt(b, a, adjusted_audio)
    
    return adjusted_audio

def apply_spectral_transfer(audio, sample_rate, target_spectrum):
    # Apply spectral transfer using Griffin-Lim algorithm
    # Compute the Short-Time Fourier Transform (STFT) of the input audio
    stft = librosa.stft(audio)
    
    # Get the magnitude and phase of the STFT
    mag, phase = librosa.magphase(stft)
    
    # Apply the target spectrum to the magnitude
    new_mag = target_spectrum * mag
    
    # Reconstruct the complex spectrum using the modified magnitude and original phase
    new_stft = new_mag * phase
    
    # Invert the STFT to get the modified audio signal
    modified_audio = librosa.istft(new_stft)
    
    return modified_audio
