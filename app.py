import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def wav_to_spectrogram(wav_file) -> tuple[np.array, int]:
    """
    args:
        wav_file: ファイルパス
    returns:
        spectrogram_list: スペクトログラム解析したリスト
        sr: サンプリングレート
    """
    # wavファイルを読み込む
    y, sr = librosa.load(wav_file)

    # -- スペクトログラムを計算 --
    # 短時間フーリエ変換
    S = librosa.stft(y)
    # 振幅スペクトログラムをデシベル単位に変換
    spectrogram_list = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    return spectrogram_list, sr

def plot_spectrogram(spectrogram_list, sr):

    # スペクトログラムをプロット
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram_list, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


# wavファイルのパスを指定
wav_file = 'tino.wav'

# スペクトログラム解析を実行
spectrogram_list, sr = wav_to_spectrogram(wav_file)



# スペクトログラムをプロット
plot_spectrogram(spectrogram_list, sr)