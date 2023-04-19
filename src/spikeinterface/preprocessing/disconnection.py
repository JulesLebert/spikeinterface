import numpy as np
from scipy.signal import welch, find_peaks

from spikeinterface.core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from ..core import get_random_data_chunks

class RemoveDisconnectionEventRecording(BasePreprocessor):
    name='remove_disconnection_event'

    def __init__(self, recording,
                 fill_value=None, compute_medians="random",
                 n_peaks=10, prominence=0.5, n_median_threshold=2,
                 num_chunks_per_segment=100, chunk_size=10000, 
                 seed=0
                 ):
        """
        Remove disconnection event from recording.
        """

        assert compute_medians in ['random', 'all']
        if compute_medians == 'random':
            subset_data = get_random_data_chunks(recording,
                                                num_chunks_per_segment=num_chunks_per_segment,
                                                chunk_size=chunk_size, seed=seed)
        elif compute_medians == 'all':
            subset_data = recording.get_traces()
        
        fs = recording.get_sampling_frequency()
        f, Pxx = welch(subset_data, fs, nperseg=1024, detrend=False, axis=0)
        Pxx_dB = np.mean(10 * np.log10(Pxx), axis=1)
        peaks, _ = find_peaks(Pxx_dB, prominence=prominence)

        if len(peaks) < n_peaks:
            median_power = None
        else:
            if compute_medians == 'all':
                num_chunks_per_segment = subset_data.shape[0] // chunk_size
                subset_data = subset_data[:num_chunks_per_segment*chunk_size, :]

            subset_data_reshaped = subset_data.reshape((num_chunks_per_segment, chunk_size, subset_data.shape[-1]))

            # power = np.sum(np.abs(random_data_reshaped)**2, axis=1)/random_data_reshaped.shape[1]
            # power = np.mean(np.square(np.abs(random_data_reshaped)), axis=1)
            power = np.mean(np.abs(subset_data_reshaped), axis=1)
            median_power = np.median(power, axis=0)

            if fill_value is None:
                fill_value = np.median(subset_data_reshaped)
        
        BasePreprocessor.__init__(self, recording)
        for parent_segment in recording._recording_segments:
            rec_segment = RemoveDisconnectionEventRecordingSegment(
                parent_segment, median_power, n_median_threshold,
                fill_value=fill_value, chunk_size=chunk_size,
                )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording, n_peaks=n_peaks, prominence=prominence, fill_value=fill_value,
                            n_median_threshold=n_median_threshold, num_chunks_per_segment=num_chunks_per_segment, 
                            chunk_size=chunk_size, seed=seed,
                        )

class RemoveDisconnectionEventRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment,
                 median_power, n_median_threshold,
                 fill_value, chunk_size=10000,
                 ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.median_power = median_power
        self.n_median_threshold = n_median_threshold
        self.fill_value = fill_value
        self.chunk_size = chunk_size

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        if self.median_power is None:
            return traces
        else:
            traces = traces.copy()
            median_power = self.median_power[channel_indices]

            chunk_powers = []
            for i in range(0, traces.shape[0], self.chunk_size):
                chunk = traces[i:i+self.chunk_size, :]
                chunk_power = np.mean(np.square(np.abs(chunk)), axis=0)
                chunk_power = np.mean(np.abs(chunk), axis=0)
                chunk_powers.append(chunk_power)
                # chunk_power = np.sum(np.abs(x)**2)/len(x)
                mask = np.greater(chunk_power, self.n_median_threshold*median_power)

                chunk[:, mask] = self.fill_value
                traces[i:i+self.chunk_size, :] = chunk
        
        chunk_powers = np.vstack(chunk_powers)

        return traces
    
remove_disconnection_event = define_function_from_class(RemoveDisconnectionEventRecording, 
                                                        name="remove_disconnection_event")