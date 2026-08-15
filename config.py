"""
游戏实时语音翻译器 - 配置模块
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """全局配置"""

    # ===== STT 设置 =====
    whisper_model: str = "base"          # tiny / base / small / medium / large-v3
    whisper_device: str = "cpu"          # cpu / cuda
    whisper_compute_type: str = "int8"   # int8 / float16 / float32

    # ===== 音频设置 =====
    target_sample_rate: int = 16000      # Whisper 要求 16kHz
    chunk_duration: float = 0.03         # 每次读取 30ms 音频

    # ===== VAD 语音活动检测 =====
    vad_energy_threshold: float = 0.015  # 能量阈值，越大越不敏感
    vad_silence_duration: float = 1.0    # 静音多少秒后判定一句话结束
    vad_min_speech_duration: float = 0.3 # 最短语音时长(秒)，短于此忽略
    vad_max_speech_duration: float = 20.0 # 最长语音时长(秒)，超过强制截断

    # ===== 翻译设置 =====
    incoming_target_lang: str = "zh"     # 听入翻译目标语言(中文)

    # ===== TTS 设置 =====
    tts_rate: str = "-5%"                # 语速 (略慢更自然)
    tts_volume: str = "+0%"              # 音量
    tts_pitch: str = "-2Hz"              # 音高 (微降更接近真人)

    # 音色选项 (按语言) - 排在前面的更拟人
    VOICE_OPTIONS = {
        "en": [
            ("Andrew (自然男声·推荐)", "en-US-AndrewMultilingualNeural"),
            ("Brian (沉稳男声)", "en-US-BrianMultilingualNeural"),
            ("Ava (自然女声)", "en-US-AvaMultilingualNeural"),
            ("Aria (清晰女声)", "en-US-AriaNeural"),
            ("Guy (浑厚男声)", "en-US-GuyNeural"),
        ],
        "ru": [
            ("Dmitry (自然男声·推荐)", "ru-RU-DmitryNeural"),
            ("Svetlana (清晰女声)", "ru-RU-SvetlanaNeural"),
        ],
    }

    # 各语言默认音色 (可在界面下拉框中切换)
    selected_voices: dict = field(default_factory=lambda: {
        "en": "en-US-AndrewMultilingualNeural",
        "ru": "ru-RU-DmitryNeural",
    })

    def get_tts_voice(self, lang: str) -> str:
        return self.selected_voices.get(lang, "en-US-AndrewMultilingualNeural")

    # ===== UI 设置 =====
    subtitle_font_size: int = 22
    subtitle_opacity: float = 0.75
    subtitle_display_duration: float = 6.0  # 字幕显示时长(秒)

    # ===== 运行时状态(不在配置文件中持久化) =====
    incoming_device_index: Optional[int] = None
    mic_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    outgoing_target_lang: str = "en"     # 说出翻译目标语言

    @property
    def chunk_size(self) -> int:
        return int(self.target_sample_rate * self.chunk_duration)

    @property
    def vad_min_chunks(self) -> int:
        """最短语音帧数"""
        return max(1, int(self.vad_min_speech_duration / self.chunk_duration))

    @property
    def vad_silence_chunks(self) -> int:
        """静音判定帧数"""
        return max(1, int(self.vad_silence_duration / self.chunk_duration))

    @property
    def vad_max_chunks(self) -> int:
        """最大语音帧数"""
        return max(1, int(self.vad_max_speech_duration / self.chunk_duration))
