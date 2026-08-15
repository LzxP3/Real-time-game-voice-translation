"""
游戏实时语音翻译器 - 主程序

功能:
  听入翻译: 捕获系统音频(游戏内外国玩家语音) → STT识别 → 翻译为中文 → 字幕显示
  说出翻译: 捕获麦克风(中文语音) → STT识别中文 → 翻译为目标语言 → TTS合成 → 播放

依赖:
  pip install faster-whisper edge-tts pyaudiowpatch numpy PyQt5 miniaudio requests
"""
import sys
import os
import time
import numpy as np
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QGroupBox, QRadioButton, QButtonGroup,
    QTextEdit, QSlider, QSpinBox, QCheckBox, QMessageBox, QFrame,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon, QTextCursor

from config import Config
from audio_manager import AudioManager, SystemAudioCapture, MicCapture, AudioPlayer
from stt_engine import STTEngine
from translator import Translator
from tts_engine import TTSEngine
from overlay import SubtitleOverlay


# ============================================================
#  工作线程
# ============================================================

class IncomingWorker(QThread):
    """听入翻译工作线程: 系统音频 → STT → 翻译 → 字幕"""

    subtitle_ready = pyqtSignal(str, str)  # (原文, 译文)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: Config, stt_engine: STTEngine, translator: Translator):
        super().__init__()
        self.config = config
        self.stt = stt_engine
        self.translator = translator
        self._running = False
        self._capture: Optional[SystemAudioCapture] = None

    def run(self):
        self._running = True
        self.status_changed.emit("正在初始化...")

        # 先检查 STT 模型是否已加载 (模型在主线程预加载，不能在 QThread 中加载)
        if not self.stt.is_loaded:
            self.error_occurred.emit("语音识别模型未加载，请等待模型加载完成")
            return

        # 再启动音频捕获
        try:
            if self.config.incoming_device_index is None:
                self.error_occurred.emit("请先选择系统音频设备")
                return

            self._capture = SystemAudioCapture(
                self.config.incoming_device_index,
                self.config.target_sample_rate,
            )
            self._capture.start()
            self.status_changed.emit(f"监听中: {self._capture.device_name}")
        except Exception as e:
            self.error_occurred.emit(f"音频设备启动失败: {e}")
            return

        self.status_changed.emit(f"监听中: {self._capture.device_name}")

        chunk_size = self.config.chunk_size
        threshold = self.config.vad_energy_threshold
        silence_limit = self.config.vad_silence_chunks
        min_speech_chunks = self.config.vad_min_chunks
        max_speech_chunks = self.config.vad_max_chunks

        audio_buffer = []
        silence_count = 0
        speech_chunk_count = 0
        is_speaking = False

        while self._running:
            chunk = self._capture.read(chunk_size)

            if len(chunk) < chunk_size:
                continue

            rms = AudioManager.calculate_rms(chunk)

            if rms > threshold:
                if not is_speaking:
                    is_speaking = True
                    audio_buffer = []
                    silence_count = 0
                    speech_chunk_count = 0
                audio_buffer.append(chunk)
                speech_chunk_count += 1

                # 超过最大时长，强制处理
                if speech_chunk_count >= max_speech_chunks:
                    self._process_audio(audio_buffer)
                    audio_buffer = []
                    is_speaking = False
                    speech_chunk_count = 0
            elif is_speaking:
                audio_buffer.append(chunk)
                silence_count += 1
                if silence_count >= silence_limit:
                    if speech_chunk_count >= min_speech_chunks:
                        self._process_audio(audio_buffer)
                    audio_buffer = []
                    is_speaking = False
                    silence_count = 0
                    speech_chunk_count = 0

        if self._capture:
            self._capture.stop()
        self.status_changed.emit("已停止")

    def _process_audio(self, buffer):
        """处理积累的音频: STT → 翻译 → 发信号"""
        if not buffer:
            return
        audio = np.concatenate(buffer)
        self.status_changed.emit("识别中...")

        text, detected_lang = self.stt.transcribe(audio)
        if not text:
            self.status_changed.emit("监听中...")
            return

        translated = self.translator.translate(text, source_lang="auto",
                                               target_lang=self.config.incoming_target_lang)
        self.subtitle_ready.emit(text, translated)
        self.status_changed.emit("监听中...")

    def stop(self):
        self._running = False


class OutgoingWorker(QThread):
    """说出翻译工作线程: 麦克风 → STT(中文) → 翻译 → TTS → 播放"""

    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    spoke_text = pyqtSignal(str, str)  # (中文原文, 翻译后外文)

    def __init__(self, config: Config, stt_engine: STTEngine,
                 translator: Translator, tts_engine: TTSEngine):
        super().__init__()
        self.config = config
        self.stt = stt_engine
        self.translator = translator
        self.tts = tts_engine
        self._running = False
        self._capture: Optional[MicCapture] = None
        self._player: Optional[AudioPlayer] = None

    def run(self):
        self._running = True
        self.status_changed.emit("正在初始化...")

        # 先检查 STT 模型是否已加载 (模型在主线程预加载，不能在 QThread 中加载)
        if not self.stt.is_loaded:
            self.error_occurred.emit("语音识别模型未加载，请等待模型加载完成")
            return

        # 再启动音频设备
        try:
            self._capture = MicCapture(
                self.config.mic_device_index,
                self.config.target_sample_rate,
            )
            self._capture.start()
            self._player = AudioPlayer(self.config.output_device_index)
            self.status_changed.emit(f"麦克风就绪: {self._capture.device_name}")
        except Exception as e:
            self.error_occurred.emit(f"音频设备启动失败: {e}")
            return

        self.status_changed.emit(f"说话就绪: {self._capture.device_name}")

        chunk_size = self.config.chunk_size
        threshold = self.config.vad_energy_threshold
        silence_limit = self.config.vad_silence_chunks
        min_speech_chunks = self.config.vad_min_chunks
        max_speech_chunks = self.config.vad_max_chunks

        audio_buffer = []
        silence_count = 0
        speech_chunk_count = 0
        is_speaking = False

        while self._running:
            chunk = self._capture.read(chunk_size)

            if len(chunk) < chunk_size:
                continue

            rms = AudioManager.calculate_rms(chunk)

            if rms > threshold:
                if not is_speaking:
                    is_speaking = True
                    audio_buffer = []
                    silence_count = 0
                    speech_chunk_count = 0
                audio_buffer.append(chunk)
                speech_chunk_count += 1

                if speech_chunk_count >= max_speech_chunks:
                    self._process_audio(audio_buffer)
                    audio_buffer = []
                    is_speaking = False
                    speech_chunk_count = 0
            elif is_speaking:
                audio_buffer.append(chunk)
                silence_count += 1
                if silence_count >= silence_limit:
                    if speech_chunk_count >= min_speech_chunks:
                        self._process_audio(audio_buffer)
                    audio_buffer = []
                    is_speaking = False
                    silence_count = 0
                    speech_chunk_count = 0

        if self._capture:
            self._capture.stop()
        self.status_changed.emit("已停止")

    def _process_audio(self, buffer):
        """处理音频: STT(中文) → 翻译 → TTS → 播放"""
        if not buffer:
            return
        audio = np.concatenate(buffer)
        self.status_changed.emit("识别中...")

        # 识别中文
        text, _ = self.stt.transcribe(audio, language="zh")
        if not text:
            self.status_changed.emit(f"说话就绪: {self._capture.device_name}")
            return

        target_lang = self.config.outgoing_target_lang
        translated = self.translator.translate(text, source_lang="zh",
                                               target_lang=target_lang)
        if not translated:
            self.status_changed.emit(f"说话就绪: {self._capture.device_name}")
            return

        self.spoke_text.emit(text, translated)
        self.status_changed.emit(f"合成语音中...")

        # TTS 合成并播放
        voice = self.config.get_tts_voice(target_lang)
        audio_data, sample_rate = self.tts.synthesize(translated, lang=target_lang,
                                                       voice=voice)

        if len(audio_data) > 0 and self._running:
            self.status_changed.emit(f"播放中...")
            self._player.play(audio_data, sample_rate)

        self.status_changed.emit(f"说话就绪: {self._capture.device_name}")

    def stop(self):
        self._running = False


# ============================================================
#  主窗口
# ============================================================

class MainWindow(QMainWindow):
    """主控窗口"""

    # 样式
    STYLE = """
        QMainWindow { background-color: #1e1e2e; }
        QGroupBox {
            color: #cdd6f4;
            font-size: 13px;
            font-weight: bold;
            border: 1px solid #45475a;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }
        QLabel { color: #cdd6f4; font-size: 12px; }
        QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #b4befe; }
        QPushButton:pressed { background-color: #74c7ec; }
        QPushButton:disabled { background-color: #45475a; color: #6c7086; }
        QPushButton[stopBtn="true"] { background-color: #f38ba8; }
        QPushButton[stopBtn="true"]:hover { background-color: #eba0ac; }
        QComboBox {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 5px 10px;
            font-size: 12px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #313244;
            color: #cdd6f4;
            selection-background-color: #89b4fa;
            selection-color: #1e1e2e;
        }
        QTextEdit {
            background-color: #181825;
            color: #a6adc8;
            border: 1px solid #313244;
            border-radius: 4px;
            font-size: 11px;
            font-family: Consolas, monospace;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #313244;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #89b4fa;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
        QRadioButton { color: #cdd6f4; font-size: 12px; }
        QCheckBox { color: #cdd6f4; font-size: 12px; }
    """

    def __init__(self, stt_engine: STTEngine = None):
        super().__init__()
        self.config = Config()

        # STT 引擎 (在 QApplication 之前已预加载，避免 ctranslate2/OpenMP 与 Qt 线程冲突)
        self.stt_engine = stt_engine or STTEngine(
            model_size=self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type,
        )
        self.translator = Translator()
        self.tts_engine = TTSEngine(
            rate=self.config.tts_rate,
            volume=self.config.tts_volume,
            pitch=self.config.tts_pitch,
        )

        # 工作线程
        self.incoming_worker: Optional[IncomingWorker] = None
        self.outgoing_worker: Optional[OutgoingWorker] = None

        # 字幕窗口
        self.overlay = SubtitleOverlay(
            font_size=self.config.subtitle_font_size,
            opacity=self.config.subtitle_opacity,
            display_duration=self.config.subtitle_display_duration,
        )

        self._init_ui()
        self._refresh_devices()

        if self.stt_engine.is_loaded:
            self._log("系统就绪，模型已加载")
            self.incoming_status.setText("状态: 就绪")
            self.outgoing_status.setText("状态: 就绪")
        else:
            self._log("警告: 模型未加载")
            self.btn_start_incoming.setEnabled(False)
            self.btn_start_outgoing.setEnabled(False)

    def _init_ui(self):
        self.setWindowTitle("游戏实时语音翻译器")
        self.setMinimumSize(640, 720)
        self.setStyleSheet(self.STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # ===== 标题 =====
        title = QLabel("🎮 游戏实时语音翻译器")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet("color: #89b4fa;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ===== 听入翻译 =====
        incoming_group = QGroupBox("📥 听入翻译 (游戏语音 → 中文字幕)")
        in_layout = QVBoxLayout(incoming_group)
        in_layout.setSpacing(6)

        # 系统音频设备
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("系统音频设备:"))
        self.incoming_device_combo = QComboBox()
        self.incoming_device_combo.setMinimumWidth(350)
        row1.addWidget(self.incoming_device_combo, 1)
        refresh_btn1 = QPushButton("🔄")
        refresh_btn1.setFixedWidth(36)
        refresh_btn1.clicked.connect(self._refresh_devices)
        row1.addWidget(refresh_btn1)
        in_layout.addLayout(row1)

        # VAD 灵敏度
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("灵敏度:"))
        self.vad_slider = QSlider(Qt.Horizontal)
        self.vad_slider.setRange(1, 100)
        self.vad_slider.setValue(15)
        self.vad_slider.valueChanged.connect(self._on_vad_changed)
        self.vad_label = QLabel("0.015")
        self.vad_label.setFixedWidth(50)
        row2.addWidget(self.vad_slider, 1)
        row2.addWidget(self.vad_label)
        in_layout.addLayout(row2)

        # 按钮
        btn_row1 = QHBoxLayout()
        self.btn_start_incoming = QPushButton("▶ 开始听入")
        self.btn_start_incoming.clicked.connect(self._start_incoming)
        self.btn_stop_incoming = QPushButton("⏹ 停止")
        self.btn_stop_incoming.setProperty("stopBtn", True)
        self.btn_stop_incoming.clicked.connect(self._stop_incoming)
        self.btn_stop_incoming.setEnabled(False)
        btn_row1.addWidget(self.btn_start_incoming)
        btn_row1.addWidget(self.btn_stop_incoming)
        in_layout.addLayout(btn_row1)

        self.incoming_status = QLabel("状态: 未启动")
        self.incoming_status.setStyleSheet("color: #6c7086;")
        in_layout.addWidget(self.incoming_status)

        layout.addWidget(incoming_group)

        # ===== 说出翻译 =====
        outgoing_group = QGroupBox("📤 说出翻译 (中文语音 → 外语输出)")
        out_layout = QVBoxLayout(outgoing_group)
        out_layout.setSpacing(6)

        # 麦克风
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("麦克风设备:"))
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(350)
        row3.addWidget(self.mic_combo, 1)
        out_layout.addLayout(row3)

        # 目标语言
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("翻译为:"))
        self.lang_en = QRadioButton("English")
        self.lang_ru = QRadioButton("Русский")
        self.lang_en.setChecked(True)
        self.lang_group = QButtonGroup()
        self.lang_group.addButton(self.lang_en)
        self.lang_group.addButton(self.lang_ru)
        self.lang_en.toggled.connect(self._on_lang_changed)
        row4.addWidget(self.lang_en)
        row4.addWidget(self.lang_ru)
        row4.addStretch()
        out_layout.addLayout(row4)

        # 音色选择
        row45 = QHBoxLayout()
        row45.addWidget(QLabel("语音音色:"))
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(350)
        self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)
        self._refresh_voice_combo()
        row45.addWidget(self.voice_combo, 1)
        out_layout.addLayout(row45)

        # 输出设备
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("语音输出到:"))
        self.output_combo = QComboBox()
        self.output_combo.setMinimumWidth(350)
        row5.addWidget(self.output_combo, 1)
        out_layout.addLayout(row5)

        # 提示
        hint = QLabel("💡 若要让游戏内外国玩家听到，请安装 VB-Cable 并选择 \"CABLE Input\" 作为输出设备")
        hint.setStyleSheet("color: #f9e2af; font-size: 11px;")
        hint.setWordWrap(True)
        out_layout.addWidget(hint)

        # 按钮
        btn_row2 = QHBoxLayout()
        self.btn_start_outgoing = QPushButton("▶ 开始说出")
        self.btn_start_outgoing.clicked.connect(self._start_outgoing)
        self.btn_stop_outgoing = QPushButton("⏹ 停止")
        self.btn_stop_outgoing.setProperty("stopBtn", True)
        self.btn_stop_outgoing.clicked.connect(self._stop_outgoing)
        self.btn_stop_outgoing.setEnabled(False)
        btn_row2.addWidget(self.btn_start_outgoing)
        btn_row2.addWidget(self.btn_stop_outgoing)
        out_layout.addLayout(btn_row2)

        self.outgoing_status = QLabel("状态: 未启动")
        self.outgoing_status.setStyleSheet("color: #6c7086;")
        out_layout.addWidget(self.outgoing_status)

        layout.addWidget(outgoing_group)

        # ===== 设置 =====
        settings_group = QGroupBox("⚙ 设置")
        s_layout = QVBoxLayout(settings_group)
        s_layout.setSpacing(6)

        row6 = QHBoxLayout()
        row6.addWidget(QLabel("Whisper模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText(self.config.whisper_model)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        row6.addWidget(self.model_combo)
        row6.addStretch()
        s_layout.addLayout(row6)

        row7 = QHBoxLayout()
        row7.addWidget(QLabel("字幕字号:"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(12, 48)
        self.font_spin.setValue(self.config.subtitle_font_size)
        self.font_spin.valueChanged.connect(self._on_font_changed)
        row7.addWidget(self.font_spin)
        row7.addStretch()
        s_layout.addLayout(row7)

        row8 = QHBoxLayout()
        self.chk_overlay = QCheckBox("显示字幕悬浮窗")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(self._on_overlay_toggle)
        row8.addWidget(self.chk_overlay)
        row8.addStretch()
        s_layout.addLayout(row8)

        layout.addWidget(settings_group)

        # ===== 日志 =====
        log_group = QGroupBox("📝 日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # 应用配置
        self._on_vad_changed(self.vad_slider.value())

    def _refresh_devices(self):
        """刷新音频设备列表"""
        self.incoming_device_combo.clear()
        self.mic_combo.clear()
        self.output_combo.clear()

        # Loopback 设备 (系统音频)
        loopback_devices = AudioManager.get_loopback_devices()
        for dev in loopback_devices:
            label = f"{dev['name']}  ({dev['hostapi']}, {dev['sample_rate']}Hz)"
            self.incoming_device_combo.addItem(label, dev["index"])

        # 输入设备 (麦克风)
        input_devices = AudioManager.get_input_devices()
        for dev in input_devices:
            label = f"{dev['name']}  ({dev['hostapi']})"
            self.mic_combo.addItem(label, dev["index"])

        # 输出设备
        output_devices = AudioManager.get_output_devices()
        for dev in output_devices:
            label = f"{dev['name']}  ({dev['hostapi']})"
            self.output_combo.addItem(label, dev["index"])

        self._log(f"设备刷新完成: {len(loopback_devices)} 个loopback设备, "
                  f"{len(input_devices)} 个麦克风, {len(output_devices)} 个输出设备")

    def _on_vad_changed(self, value):
        threshold = value / 1000.0
        self.config.vad_energy_threshold = threshold
        self.vad_label.setText(f"{threshold:.3f}")

    def _on_lang_changed(self):
        if self.lang_en.isChecked():
            self.config.outgoing_target_lang = "en"
        else:
            self.config.outgoing_target_lang = "ru"
        self._refresh_voice_combo()

    def _refresh_voice_combo(self):
        """根据当前目标语言刷新音色下拉框"""
        lang = self.config.outgoing_target_lang
        options = self.config.VOICE_OPTIONS.get(lang, [])
        current = self.config.selected_voices.get(lang, "")

        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for label, voice_id in options:
            self.voice_combo.addItem(label, userData=voice_id)
        # 选中当前配置的音色
        idx = next((i for i, (_, v) in enumerate(options) if v == current), 0)
        if self.voice_combo.count() > 0:
            self.voice_combo.setCurrentIndex(idx)
        self.voice_combo.blockSignals(False)

    def _on_voice_changed(self, index):
        voice_id = self.voice_combo.itemData(index)
        if voice_id:
            lang = self.config.outgoing_target_lang
            self.config.selected_voices[lang] = voice_id
            label = self.voice_combo.currentText()
            self._log(f"音色切换为: {label}")

    def _on_model_changed(self, text):
        self.config.whisper_model = text
        self._log(f"模型切换为 {text} 需要重启程序才能生效 (修改 config.py 中的 whisper_model)")

    def _on_font_changed(self, size):
        self.config.subtitle_font_size = size
        self.overlay.update_font_size(size)

    def _on_overlay_toggle(self, state):
        if state == Qt.Checked.value:
            self.overlay.show()
        else:
            self.overlay.hide()

    # ===== 听入控制 =====

    def _start_incoming(self):
        if self.incoming_device_combo.count() == 0:
            QMessageBox.warning(self, "提示", "未找到系统音频设备，请检查音频设置")
            return

        self.config.incoming_device_index = self.incoming_device_combo.currentData()

        self.incoming_worker = IncomingWorker(self.config, self.stt_engine, self.translator)
        self.incoming_worker.subtitle_ready.connect(self._on_incoming_subtitle)
        self.incoming_worker.status_changed.connect(self._on_incoming_status)
        self.incoming_worker.error_occurred.connect(self._on_incoming_error)
        self.incoming_worker.finished.connect(self._on_incoming_finished)
        self.incoming_worker.start()

        self.btn_start_incoming.setEnabled(False)
        self.btn_stop_incoming.setEnabled(True)
        self.incoming_device_combo.setEnabled(False)

    def _stop_incoming(self):
        if self.incoming_worker:
            self.incoming_worker.stop()
            self._log("正在停止听入翻译...")

    def _on_incoming_subtitle(self, original, translated):
        if self.chk_overlay.isChecked():
            self.overlay.show_subtitle(original, translated)
        self._log(f"[听入] {original} → {translated}")

    def _on_incoming_status(self, status):
        self.incoming_status.setText(f"状态: {status}")

    def _on_incoming_error(self, error):
        self._log(f"[听入错误] {error}")
        QMessageBox.critical(self, "错误", error)
        self._on_incoming_finished()

    def _on_incoming_finished(self):
        self.btn_start_incoming.setEnabled(True)
        self.btn_stop_incoming.setEnabled(False)
        self.incoming_device_combo.setEnabled(True)
        self.incoming_status.setText("状态: 已停止")

    # ===== 说出控制 =====

    def _start_outgoing(self):
        if self.mic_combo.count() == 0:
            QMessageBox.warning(self, "提示", "未找到麦克风设备")
            return

        self.config.mic_device_index = self.mic_combo.currentData()
        self.config.output_device_index = self.output_combo.currentData()
        self._on_lang_changed()

        self.outgoing_worker = OutgoingWorker(
            self.config, self.stt_engine, self.translator, self.tts_engine
        )
        self.outgoing_worker.status_changed.connect(self._on_outgoing_status)
        self.outgoing_worker.error_occurred.connect(self._on_outgoing_error)
        self.outgoing_worker.spoke_text.connect(self._on_spoke_text)
        self.outgoing_worker.finished.connect(self._on_outgoing_finished)
        self.outgoing_worker.start()

        self.btn_start_outgoing.setEnabled(False)
        self.btn_stop_outgoing.setEnabled(True)
        self.mic_combo.setEnabled(False)
        self.output_combo.setEnabled(False)

    def _stop_outgoing(self):
        if self.outgoing_worker:
            self.outgoing_worker.stop()
            self._log("正在停止说出翻译...")

    def _on_outgoing_status(self, status):
        self.outgoing_status.setText(f"状态: {status}")

    def _on_outgoing_error(self, error):
        self._log(f"[说出错误] {error}")
        QMessageBox.critical(self, "错误", error)
        self._on_outgoing_finished()

    def _on_spoke_text(self, original, translated):
        self._log(f"[说出] {original} → {translated}")

    def _on_outgoing_finished(self):
        self.btn_start_outgoing.setEnabled(True)
        self.btn_stop_outgoing.setEnabled(False)
        self.mic_combo.setEnabled(True)
        self.output_combo.setEnabled(True)
        self.outgoing_status.setText("状态: 已停止")

    # ===== 工具 =====

    def _log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        self.log_text.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        """关闭时清理"""
        if self.incoming_worker:
            self.incoming_worker.stop()
            self.incoming_worker.wait(3000)
        if self.outgoing_worker:
            self.outgoing_worker.stop()
            self.outgoing_worker.wait(3000)
        self.overlay.close()
        event.accept()


# ============================================================
#  入口
# ============================================================

# !! 重要 !!
# 本文件顶部导入了 PyQt5，而 ctranslate2 的模型加载必须在 Qt DLL
# 加载之前完成，否则段错误。因此不要直接运行 main.py，
# 请通过 launcher.py (或 run.bat) 启动——它会先加载模型再导入本模块。

if __name__ == "__main__":
    print("请勿直接运行 main.py (会导致段错误)。", flush=True)
    print("请运行 launcher.py 或双击 run.bat 启动。", flush=True)
    import subprocess
    subprocess.call([sys.executable, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "launcher.py")])


if __name__ == "__main__":
    main()
