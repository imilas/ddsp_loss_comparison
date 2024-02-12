import soundfile as sf
import io
from six.moves.urllib.request import urlopen

def load_url(url,mono=True):
    y,sr = sf.read(io.BytesIO(urlopen(url).read()))
    if mono:
        return y.mean(axis=1),sr
    else:
        return y,sr
