import torchaudio
from torchaudio import transforms
import torch
import random


class SoundLoader:
    def __init__(
        self,
        feat_ext=torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH.get_streaming_feature_extractor(),
        shift_limit=0.4,
        sr=16000,
        max_ms=1000,
        channel=1,
    ):
        self.sr = sr
        self.feat_ext = feat_ext
        self.max_ms = max_ms
        self.channel = channel
        self.shift_limit = shift_limit

    def open(self, audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    def load(self, audio_file):
        sig, sr = self.open(audio_file)
        sig, sr = self.resample((sig, sr))
        sig, sr = self.rechannel((sig, sr))
        feat, _ = self.feat_ext(sig.squeeze(0))
        return feat

    def rechannel(self, aud):
        sig, sr = aud
        if sig.shape[0] == self.channel:
            return aud
        if self.channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])
        return (resig, sr)

    def resample(self, aud):
        sig, sr = aud
        if sr == self.sr:
            return aud
        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, self.sr)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, self.sr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
        return (resig, self.sr)

    def pad_trunc(self, aud):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * self.max_ms

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    def time_shift(self, aud):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * self.shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    def spectro_gram(self, aud, hop_len=None):
        sig, sr = aud
        spec = transforms.MelSpectrogram(sr, n_fft=1024, n_mels=64)(sig)
        spec = transforms.AmplitudeToDB(top_db=80)(spec)
        return spec

    def spectro_augment(self, spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(
                aug_spec, mask_value
            )

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec


if __name__ == "__main__":
    print("Loading data...")
