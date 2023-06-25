import torch
import torchaudio
import sounddevice as sd
from utils import ContextCacher
from model.head import RawHead
from tts import play_wtf
from googletrans import Translator


# The data acquisition process will stop after this number of steps.
# This eliminates the need of process synchronization and makes this
# tutorial simple.
NUM_ITER = 1000
THRESHOLD = 0.62855


class Pipeline:
    """Build inference pipeline from RNNTBundle.

    Args:
        bundle (torchaudio.pipelines.RNNTBundle): Bundle object
        beam_width (int): Beam size of beam search decoder.
    """

    def __init__(self, bundle: torchaudio.pipelines.RNNTBundle, detector, beam_width: int = 10):
        self.bundle = bundle
        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.detector = detector
        self.token_processor = bundle.get_token_processor()

        self.beam_width = beam_width
        self.translator = Translator()

        self.state = None
        self.c_state = None
        self.hypothesis = None
        self.transcript = ''
        self.last_word = None
        self.blocking = False

    def infer(self, segment: torch.Tensor) -> str:
        """Perform streaming inference"""
        features, length = self.feature_extractor(segment)
        _, conf = self.detector(features)
        hypos, self.state = self.decoder.infer(
            features,
            length,
            self.beam_width,
            state=self.state,
            hypothesis=self.hypothesis,
        )   
        self.hypothesis = hypos[0]
        self.transcript = self.token_processor(self.hypothesis[0], lstrip=False)
        lastword = self.token_processor(self.hypothesis[0], lstrip=True)
        if conf.sigmoid().mean().item() > THRESHOLD and lastword != self.last_word and lastword not in  [""] and not self.blocking:
            translated = self.translator.translate(lastword, dest="th").text
            print("\033[91m {} {} \033[00m" .format(lastword,translated))
            self.blocking = True
            play_wtf(translated)
            self.blocking = False
        
        self.last_word = lastword if lastword not in ["", " "] else self.last_word
        return self.transcript


def main(bundle,weigth=None):
    print(torch.__version__)
    print(torchaudio.__version__)

    print("Building pipeline...")
    head = RawHead(80,4,1)
    if weigth:
        head = torch.load(weigth)
    pipeline = Pipeline(bundle,head)

    sample_rate = bundle.sample_rate
    segment_length = bundle.segment_length * bundle.hop_length
    context_length = bundle.right_context_length * bundle.hop_length

    print(f"Sample rate: {sample_rate}")
    print(
        f"Main segment: {segment_length} frames ({segment_length / sample_rate} seconds)"
    )
    print(
        f"Right context: {context_length} frames ({context_length / sample_rate} seconds)"
    )

    cacher = ContextCacher(segment_length, context_length)

    @torch.inference_mode()
    def infer():
        for _ in range(NUM_ITER):
            chunk = torch.Tensor(q.get())
            segment = cacher(chunk[:, 0])
            transcript = pipeline.infer(segment)
            print(transcript, end="", flush=True)

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=stream, args=(q, segment_length, sample_rate))
    p.start()
    infer()
    p.join()


def stream(q, segment_length, sample_rate):
    for _ in range(NUM_ITER):
      chunk = sd.rec(segment_length,sample_rate,channels=1,blocking=True)
      q.put(chunk)

          


if __name__ == "__main__":
    print('Started....')
    # play_wtf('นวัตกรรมเปลี่ยนแปลงโลก')
    main(
        torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH,
        './models/model-100.pth'
    )
