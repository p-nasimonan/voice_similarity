import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sounddevice as sd
import speech_recognition as sr_recog
from pydub import AudioSegment
from pydub.silence import split_on_silence


# 日本語フォントを指定
jp_font_path = 'font/meiryo.ttc'
jp_font = fm.FontProperties(fname=jp_font_path)


def audio_to_spectrogram(audio_path: str) -> tuple[np.ndarray, int, int, int]:
    """
    音声ファイルを読み込んでスペクトログラムを計算する関数

    Args:
        audio_path (str): 音声ファイルのパス

    Returns:
        np.ndarray: スペクトログラム
        sr: サンプリングレート
        n_fft: フレームサイズ
        hop_length: ホップサイズ
    """
    y, sr = librosa.load(audio_path)

    # 音声データの長さに基づいてn_fftとhop_lengthを設定
    n_fft = int(0.025 * sr)  # 25msのフレームサイズ
    hop_length = int(0.010 * sr)  # 10msのホップサイズ
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    return spectrogram, sr, n_fft, hop_length


def recognize_speech(audio_path: str) -> str:
    """
    音声ファイルを音声認識する関数

    Args:
        audio_path (str): 音声ファイルのパス

    Returns:
        str: 音声認識されたテキスト
    """
    recognizer = sr_recog.Recognizer()
    with sr_recog.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ja-JP")
            return text
        except sr_recog.UnknownValueError:
            return "[Unrecognized]"
        except sr_recog.RequestError as e:
            return f"[Error: {e}]"
        


def split_audio_by_characters(audio_path, sr=22050):
    """
    音声ファイルを文字ごとに分割する関数

    Args:
        audio_path (str): 音声ファイルのパス
        text (str): 音声ファイルに含まれるテキスト
        sr (int, optional): サンプリングレート。デフォルトは22050Hz。

    Returns:
        list[dict]: 文字ごとに分割された音声ファイルのリスト
    """

    # 無音部分で音声を分割
    audio = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(audio, min_silence_len=10000, silence_thresh=-50)
    
    recognizer = sr_recog.Recognizer()
    char_times = []
    current_time = 0.0

    temp_chunk_path = "temp/chunk.wav"
    
    for chunk in chunks:
        chunk.export(temp_chunk_path, format="wav")
        
        with sr_recog.AudioFile(temp_chunk_path) as source:
            audio_data = recognizer.record(source)
            y, sr = librosa.load(temp_chunk_path, sr=sr)
            chunk_duration = librosa.get_duration(y=y, sr=sr)
            try:
                chunk_text = recognizer.recognize_google(audio_data, language="ja-JP")
                for char in chunk_text:
                    char_times.append({char: current_time})
                    current_time += chunk_duration / len(chunk_text)
            except sr_recog.UnknownValueError:
                char_times.append({" ": current_time})
                current_time += chunk_duration
            except sr_recog.RequestError as e:
                char_times.append({f"[Error: {e}]": current_time})

    
    return char_times





def spectrogram_to_audio(spectrogram: np.ndarray, sr: int, n_fft: int, hop_length: int):
    """
    スペクトログラムのリストを音声信号に変換して再生する関数

    Args:
        spectrogram_list (list): スペクトログラムのリスト表現
        sr (int, optional): サンプリングレート。デフォルトは22050Hz。
        hop_length (int, optional): ホップサイズ。デフォルトは512。
    """ 
    y = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr, n_fft=n_fft, hop_length=hop_length)
    sd.play(y, sr)
    sd.wait()


def plot_spectrogram(spectrogram: np.ndarray, sr: int):
    # デシベルスケールに変換
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    # スペクトログラムをプロット
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

def plot_spectrogram_with_text(spectrogram: np.ndarray, sr: int, n_fft: int, hop_length: int, char_times: list):

    # デシベルスケールに変換
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectrogram_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    # 文字とその時間をプロット
    fontsize=12
    for char_time in char_times:
        for char, time in char_time.items():
            plt.text(time, fontsize, char, color='white', fontsize=fontsize, ha='center', va='center', fontproperties=jp_font)
    
    plt.show()



def main():
    """
    メイン関数
    """
    audio_path = "tino.wav"  # 音声ファイルのパスを指定
    spectrogram, sr, n_fft, hop_length = audio_to_spectrogram(audio_path)
    
    # 音声全体を音声認識
    recognized_text = recognize_speech(audio_path)
    print(f"Recognized Text: {recognized_text}")
    
    # 認識された文字に対応する時間を測定
    char_times = split_audio_by_characters(audio_path, sr)
    print("Character Times:", char_times)

    # スペクトログラムと文字をプロット
    plot_spectrogram_with_text(spectrogram, sr, n_fft, hop_length, char_times)

    

if __name__ == "__main__":
    main()