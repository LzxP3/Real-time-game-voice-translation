"""
语音合成引擎 (TTS) - 基于 edge-tts (微软 Edge 在线 TTS)
"""
import asyncio
import numpy as np
from typing import Tuple


class TTSEngine:
    """文本转语音"""

    def __init__(self, rate: str = "-5%", volume: str = "+0%", pitch: str = "-2Hz"):
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self._loop = None

    @staticmethod
    def _prepare_text(text: str, lang: str) -> str:
        """
        文本韵律预处理: 标点决定 TTS 的停顿和语调，
        机器翻译输出的文本经常缺少标点，会导致一字一顿的机器人腔。
        """
        text = text.strip()
        if not text:
            return text

        # 句尾无标点时补句号 (陈述语气，避免上扬的疑问语调)
        terminal = tuple("。．.！!？?…")
        if not text.endswith(terminal):
            text += "." if lang != "zh" else "。"

        # 超长且无任何标点的文本，按 ~12 字/词 插入逗号制造自然停顿
        if not any(c in text for c in "，,。.！!？?；;、"):
            if lang == "zh":
                words = [text[i:i + 12] for i in range(0, len(text), 12)]
            else:
                tokens = text.split(" ")
                words = [" ".join(tokens[i:i + 8]) for i in range(0, len(tokens), 8)]
            text = "，".join(words) if lang == "zh" else ", ".join(words)

        return text

    def _get_loop(self):
        """获取或创建事件循环(用于在 QThread 中运行 async 代码)"""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    async def _synthesize_async(self, text: str, voice: str) -> bytes:
        """异步合成语音，返回 MP3 字节数据"""
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )

        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data

    def synthesize(self, text: str, lang: str = "en",
                   voice: str = None) -> Tuple[np.ndarray, int]:
        """
        同步合成语音

        Args:
            text: 待合成文本
            lang: 语言代码
            voice: 指定语音名称(可选), 为 None 时用默认语音

        Returns:
            (audio_data: float32 numpy数组, sample_rate: int)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), 24000

        if voice is None:
            voice = self._get_default_voice(lang)

        try:
            loop = self._get_loop()
            prepared = self._prepare_text(text, lang)
            mp3_data = loop.run_until_complete(
                self._synthesize_async(prepared, voice)
            )

            if len(mp3_data) == 0:
                return np.array([], dtype=np.float32), 24000

            # 用 miniaudio 解码 MP3
            import miniaudio
            decoded = miniaudio.decode(
                mp3_data,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=1,
                sample_rate=24000,
            )
            audio_array = np.frombuffer(decoded.samples, dtype=np.float32)
            return audio_array, decoded.sample_rate

        except Exception as e:
            print(f"[TTS] 合成失败: {e}")
            return np.array([], dtype=np.float32), 24000

    def _get_default_voice(self, lang: str) -> str:
        voices = {
            "en": "en-US-AndrewMultilingualNeural",
            "ru": "ru-RU-DmitryNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
        }
        return voices.get(lang, "en-US-AndrewMultilingualNeural")
