"""
音频管理模块 (全部使用 pyaudiowpatch，避免与 ctranslate2 的线程冲突)
- 系统音频捕获 (WASAPI Loopback)
- 麦克风捕获
- VAD 语音活动检测
- 音频重采样与格式转换
- 音频播放
"""
import numpy as np
from typing import Optional, List, Dict


class AudioManager:
    """音频设备管理与工具方法"""

    @staticmethod
    def _get_pyaudio():
        """创建并返回 PyAudio 实例"""
        import pyaudiowpatch as pyaudio
        return pyaudio.PyAudio()

    @staticmethod
    def get_loopback_devices() -> List[Dict]:
        """获取可用于 loopback 捕获的系统音频设备列表"""
        devices = []
        try:
            p = AudioManager._get_pyaudio()
            for dev in p.get_loopback_device_info_generator():
                devices.append({
                    "index": dev["index"],
                    "name": dev["name"],
                    "channels": dev["maxInputChannels"],
                    "sample_rate": int(dev["defaultSampleRate"]),
                    "hostapi": "WASAPI Loopback",
                })
            p.terminate()
        except ImportError:
            print("[AudioManager] pyaudiowpatch 未安装")
        except Exception as e:
            print(f"[AudioManager] 获取 loopback 设备失败: {e}")
        return devices

    @staticmethod
    def get_input_devices() -> List[Dict]:
        """获取麦克风/输入设备列表 (排除 loopback 设备)"""
        devices = []
        try:
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            api_count = p.get_host_api_count()
            api_names = {}
            for i in range(api_count):
                api_info = p.get_host_api_info_by_index(i)
                api_names[i] = api_info["name"]

            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    name = info["name"]
                    # 排除 loopback 设备 (它们的名字包含 [Loopback])
                    if "[Loopback]" in name or "Loopback" in name:
                        continue
                    hostapi = api_names.get(info.get("hostApi", 0), "Unknown")
                    devices.append({
                        "index": i,
                        "name": name,
                        "channels": int(info["maxInputChannels"]),
                        "sample_rate": int(info["defaultSampleRate"]),
                        "hostapi": hostapi,
                    })
            p.terminate()
        except Exception as e:
            print(f"[AudioManager] 获取输入设备失败: {e}")
        return devices

    @staticmethod
    def get_output_devices() -> List[Dict]:
        """获取输出/扬声器设备列表"""
        devices = []
        try:
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            api_count = p.get_host_api_count()
            api_names = {}
            for i in range(api_count):
                api_info = p.get_host_api_info_by_index(i)
                api_names[i] = api_info["name"]

            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info["maxOutputChannels"] > 0:
                    hostapi = api_names.get(info.get("hostApi", 0), "Unknown")
                    devices.append({
                        "index": i,
                        "name": info["name"],
                        "channels": int(info["maxOutputChannels"]),
                        "sample_rate": int(info["defaultSampleRate"]),
                        "hostapi": hostapi,
                    })
            p.terminate()
        except Exception as e:
            print(f"[AudioManager] 获取输出设备失败: {e}")
        return devices

    @staticmethod
    def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        """线性插值重采样"""
        if source_rate == target_rate:
            return audio
        duration = len(audio) / source_rate
        target_length = int(duration * target_rate)
        if target_length == 0:
            return np.array([], dtype=np.float32)
        source_indices = np.arange(len(audio))
        target_indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(target_indices, source_indices, audio).astype(np.float32)

    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        """多声道转单声道"""
        if audio.ndim > 1:
            return audio.mean(axis=1).astype(np.float32)
        return audio.astype(np.float32)

    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        """计算 RMS 能量"""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


class SystemAudioCapture:
    """通过 pyaudiowpatch WASAPI Loopback 捕获系统音频"""

    def __init__(self, device_index: int, sample_rate: int = 16000):
        self.device_index = device_index
        self.target_sample_rate = sample_rate
        self._pyaudio = None
        self._stream = None
        self._native_rate = 48000
        self._native_channels = 2
        self.device_name = f"Device {device_index}"

        # 查询设备信息
        try:
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            info = p.get_device_info_by_index(device_index)
            self._native_rate = int(info["defaultSampleRate"])
            self._native_channels = int(info["maxInputChannels"])
            self.device_name = info["name"]
            p.terminate()
        except Exception as e:
            print(f"[SystemAudioCapture] 获取设备信息失败: {e}")

    def start(self):
        """打开音频流"""
        import pyaudiowpatch as pyaudio

        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        try:
            self._stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self._native_channels,
                rate=self._native_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=int(self._native_rate * 0.03),
            )
        except Exception as e:
            raise RuntimeError(f"无法打开系统音频设备: {e}")

    def read(self, frames: int) -> np.ndarray:
        """读取音频数据，返回目标采样率单声道 float32"""
        if self._stream is None:
            return np.zeros(frames, dtype=np.float32)

        try:
            native_frames = int(frames * self._native_rate / self.target_sample_rate)
            raw_data = self._stream.read(native_frames, exception_on_overflow=False)
            data = np.frombuffer(raw_data, dtype=np.float32)
        except Exception as e:
            print(f"[SystemAudioCapture] 读取失败: {e}")
            return np.zeros(frames, dtype=np.float32)

        # 多声道转单声道
        if self._native_channels > 1:
            data = data.reshape(-1, self._native_channels).mean(axis=1)

        # 重采样
        resampled = AudioManager.resample(data, self._native_rate, self.target_sample_rate)
        return resampled

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None


class MicCapture:
    """麦克风音频捕获 (使用 pyaudiowpatch)"""

    def __init__(self, device_index: Optional[int] = None, sample_rate: int = 16000):
        self.device_index = device_index
        self.target_sample_rate = sample_rate
        self._pyaudio = None
        self._stream = None
        self._native_rate = 16000
        self._native_channels = 1
        self.device_name = "默认麦克风"

        try:
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            if device_index is not None:
                info = p.get_device_info_by_index(device_index)
                self._native_rate = int(info["defaultSampleRate"])
                self._native_channels = max(1, int(info["maxInputChannels"]))
                self.device_name = info["name"]
            else:
                default_input = p.get_default_input_device_info()
                self._native_rate = int(default_input["defaultSampleRate"])
                self._native_channels = max(1, int(default_input["maxInputChannels"]))
                self.device_name = default_input["name"]
                self.device_index = default_input["index"]
            p.terminate()
        except Exception as e:
            print(f"[MicCapture] 获取设备信息失败: {e}")

    def start(self):
        import pyaudiowpatch as pyaudio

        try:
            if self._pyaudio is None:
                self._pyaudio = pyaudio.PyAudio()

            self._stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self._native_channels,
                rate=self._native_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=int(self._native_rate * 0.03),
            )
        except Exception as e:
            raise RuntimeError(f"无法打开麦克风: {e}")

    def read(self, frames: int) -> np.ndarray:
        if self._stream is None:
            return np.zeros(frames, dtype=np.float32)

        try:
            native_frames = int(frames * self._native_rate / self.target_sample_rate)
            raw_data = self._stream.read(native_frames, exception_on_overflow=False)
            data = np.frombuffer(raw_data, dtype=np.float32)
        except Exception as e:
            print(f"[MicCapture] 读取失败: {e}")
            return np.zeros(frames, dtype=np.float32)

        if self._native_channels > 1:
            data = data.reshape(-1, self._native_channels).mean(axis=1)

        resampled = AudioManager.resample(data, self._native_rate, self.target_sample_rate)
        return resampled

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None


class AudioPlayer:
    """音频播放器 (使用 pyaudiowpatch)"""

    def __init__(self, device_index: Optional[int] = None):
        self.device_index = device_index

    def play(self, audio_data: np.ndarray, sample_rate: int):
        """同步播放音频数据"""
        import pyaudiowpatch as pyaudio

        _pyaudio = None
        _stream = None
        try:
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float32)

            _pyaudio = pyaudio.PyAudio()
            _stream = _pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=int(sample_rate),
                output=True,
                output_device_index=self.device_index,
            )
            _stream.write(audio_data.tobytes())
            _stream.stop_stream()
        except Exception as e:
            print(f"[AudioPlayer] 播放失败: {e}")
        finally:
            if _stream is not None:
                try:
                    _stream.close()
                except Exception:
                    pass
            if _pyaudio is not None:
                try:
                    _pyaudio.terminate()
                except Exception:
                    pass
