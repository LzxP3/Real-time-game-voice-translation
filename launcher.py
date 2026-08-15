"""
启动器 (launcher.py) - 唯一正确的入口

关键约束: ctranslate2 (faster-whisper 底层) 的 OpenMP 线程池必须在
PyQt5 的 Qt DLL 被加载进进程之前初始化完成，
否则两者线程模型冲突会导致段错误 (Segmentation fault)，程序直接消失。

因此本文件在导入任何 Qt 模块之前先完成模型加载。
请始终通过本文件或 run.bat 启动程序，不要直接运行 main.py。
"""
import os
import sys

# 确保能找到项目内模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 国内镜像 (后备，模型已在本地时不会联网)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main():
    # ---- 第 1 步: 在导入 Qt 之前加载 ctranslate2 模型 ----
    print("[启动] 正在加载语音识别模型...", flush=True)
    try:
        from config import Config
        from stt_engine import STTEngine

        config = Config()
        stt_engine = STTEngine(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )
        stt_engine.load_model()
    except Exception as e:
        import traceback
        traceback.print_exc()
        with open("crash_log.txt", "w", encoding="utf-8") as f:
            f.write(f"模型加载失败:\n{traceback.format_exc()}")
        input(f"\n[错误] 模型加载失败: {e}\n按回车键退出...")
        sys.exit(1)
    print("[启动] 模型加载完成", flush=True)

    # ---- 第 2 步: 模型就绪后才能导入 Qt / UI 模块 ----
    from PyQt5.QtWidgets import QApplication
    import main as app_module

    app = QApplication(sys.argv)
    app.setApplicationName("GameVoiceTranslator")

    window = app_module.MainWindow(stt_engine=stt_engine)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
