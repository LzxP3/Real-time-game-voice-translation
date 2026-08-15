"""
字幕悬浮窗 - 透明置顶窗口，显示翻译字幕
"""
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QPointF
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QPen, QPainterPath
from PyQt5.QtGui import QLinearGradient, QBrush, QFontMetrics


class SubtitleOverlay(QWidget):
    """透明置顶字幕窗口"""

    def __init__(self, font_size: int = 22, opacity: float = 0.75,
                 display_duration: float = 6.0):
        super().__init__()

        self.font_size = font_size
        self.opacity = opacity
        self.display_duration = display_duration

        # 窗口属性: 无边框、置顶、半透明、接受鼠标
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        # 布局
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(24, 16, 24, 16)
        self._layout.setSpacing(4)

        # 原文标签 (小字, 灰色)
        self.original_label = QLabel("")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFont(QFont("Microsoft YaHei", max(12, font_size - 6)))
        self.original_label.setStyleSheet("color: rgba(200, 200, 200, 230);")
        self._layout.addWidget(self.original_label)

        # 译文标签 (大字, 白色)
        self.translated_label = QLabel("")
        self.translated_label.setAlignment(Qt.AlignCenter)
        self.translated_label.setFont(QFont("Microsoft YaHei", font_size, QFont.Bold))
        self.translated_label.setStyleSheet("color: white;")
        self._layout.addWidget(self.translated_label)

        # 自动隐藏定时器
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

        # 淡出动画
        self._fade_anim = None

        # 可拖动
        self._drag_pos = None

        self.setFixedSize(700, 140)
        self._position_bottom_center()

    def _position_bottom_center(self):
        """定位到屏幕底部中央"""
        screen = self.screen().geometry() if self.screen() else None
        if screen is None:
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen().geometry()
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + screen.height() - self.height() - 80
        self.move(x, y)

    def paintEvent(self, event):
        """绘制半透明背景"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 圆角矩形背景
        path = QPainterPath()
        path.addRoundedRect(0.0, 0.0, self.width(), self.height(), 16.0, 16.0)

        # 渐变背景
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(0, 0, 0, int(200 * self.opacity)))
        gradient.setColorAt(1.0, QColor(0, 0, 0, int(180 * self.opacity)))
        painter.fillPath(path, QBrush(gradient))

        # 边框
        pen = QPen(QColor(255, 255, 255, 30))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPath(path)

    def show_subtitle(self, original: str, translated: str):
        """显示字幕"""
        # 取消隐藏定时器和动画
        self._hide_timer.stop()
        self.setWindowOpacity(1.0)

        # 设置文本
        self.original_label.setText(original)
        self.translated_label.setText(translated)

        # 根据文本长度自适应宽度
        fm = QFontMetrics(self.translated_label.font())
        text_width = fm.horizontalAdvance(translated)
        original_fm = QFontMetrics(self.original_label.font())
        original_width = original_fm.horizontalAdvance(original)
        max_width = max(text_width, original_width)

        new_width = min(max(max_width + 60, 300), 900)
        self.setFixedSize(new_width, 140)
        self._position_bottom_center()

        self.show()
        self.raise_()

        # 启动自动隐藏
        self._hide_timer.start(int(self.display_duration * 1000))

    def _fade_out(self):
        """淡出动画"""
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(500)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.finished.connect(self.hide)
        self._fade_anim.start()

    def mousePressEvent(self, event):
        """开始拖动"""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """拖动移动"""
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        """结束拖动"""
        self._drag_pos = None

    def update_font_size(self, size: int):
        self.font_size = size
        self.original_label.setFont(QFont("Microsoft YaHei", max(12, size - 6)))
        self.translated_label.setFont(QFont("Microsoft YaHei", size, QFont.Bold))
