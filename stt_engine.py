"""
语音识别引擎 (STT) - 基于 faster-whisper
"""
import os
# 必须在导入 faster_whisper 之前设置国内镜像，否则模型下载会超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 禁用 Xet 协议，强制 HTTP 下载

import numpy as np
from typing import Optional, Tuple


class STTEngine:
    """使用 faster-whisper 进行语音识别"""

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load_model(self):
        """加载 Whisper 模型（优先本地目录，避免联网下载）"""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        # 优先从本地 models/<model_size>/ 目录加载
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", self.model_size
        )
        model_path = local_path if os.path.isdir(local_path) else self.model_size

        print(f"[STT] 正在加载模型: {self.model_size} -> {model_path} ({self.device}/{self.compute_type})...")
        self._model = WhisperModel(
            model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        print(f"[STT] 模型加载完成")

    def transcribe(self, audio_data: np.ndarray,
                   language: Optional[str] = None) -> Tuple[str, str]:
        """
        识别音频中的语音

        Args:
            audio_data: float32 numpy 数组, 16kHz 单声道
            language: 强制语言代码 (如 "zh", "en", "ru"), None 则自动检测

        Returns:
            (识别文本, 检测到的语言代码)
        """
        if self._model is None:
            self.load_model()

        if len(audio_data) == 0:
            return "", ""

        # faster-whisper 需要 float32, 16kHz
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        try:
            segments, info = self._model.transcribe(
                audio_data,
                language=language,
                beam_size=1,          # 贪心解码，速度最快
                best_of=1,
                temperature=0.0,      # 确定性输出
                vad_filter=True,      # 内置 VAD 过滤静音
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
            )
            text = " ".join([seg.text.strip() for seg in segments]).strip()
            return text, info.language
        except Exception as e:
            print(f"[STT] 识别失败: {e}")
            return "", ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
