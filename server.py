import base64
import io

import litserve as ls
import torch
import torchaudio

from urbansound.multi_branch_lightning import LitMultiBranchFusion


class AudioAPI(ls.LitAPI):
    def setup(self, device):
        self.model = LitMultiBranchFusion.load_from_checkpoint(
            "/teamspace/studios/this_studio/urbansound/multi_branch_checkpoints/mb-fold-10-best-v1.ckpt"
        )
        self.class_mapping = [
            "air_conditioner",
            "car_horn",
            "children_playing",
            "dog_bark",
            "drilling",
            "engine_idling",
            "gun_shot",
            "jackhammer",
            "siren",
            "street_music",
        ]
        self.target_sample_rate = 22050
        self.num_samples = 22050

    def decode_request(self, request):
        b64_audios = request["audio_b64"]
        if not isinstance(b64_audios, list):
            b64_audios = [b64_audios]

        processed_signals = []
        for b64_audio in b64_audios:
            audio_bytes = base64.b64decode(b64_audio)
            signal, sr = torchaudio.load(io.BytesIO(audio_bytes))

            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)

            processed_signals.append(signal)

        # Stack into [Batch, Channels, Time] e.g., [N, 1, 22050]
        batched_signals = torch.stack(processed_signals)
        return batched_signals

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=self.target_sample_rate
        )
        if sr != self.target_sample_rate:
            # resampler = resampler.to(self.device) # Does not work if not brought to device
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            mel_spec, mfcc, global_stats = self.model.extract_features(
                x.to(self.model.device)
            )
            predictions, alpha = self.model(mel_spec, mfcc, global_stats)

            results = []
            for pred in predictions:
                predicted_index = pred.argmax(0)
                predicted_class = self.class_mapping[predicted_index]
                results.append(predicted_class)

        return {"output": results}

    def encode_response(self, output):
        return {"output": output["output"]}


if __name__ == "__main__":
    api = AudioAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)
