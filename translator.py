"""
翻译引擎 - 使用 MyMemory 免费 API (国内可用)
备选: deep-translator GoogleTranslator (需翻墙)
"""
import requests
import time
from typing import Optional


class Translator:
    """文本翻译 - MyMemory API"""

    # 语言代码映射
    LANG_MAP = {
        "zh": "zh-CN",
        "en": "en",
        "ru": "ru",
        "auto": "Autodetect",
    }

    def __init__(self):
        self._cache = {}
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self._last_request_time = 0
        self._min_interval = 0.5  # 最小请求间隔(秒)，避免被限流

    def translate(self, text: str, source_lang: str = "auto",
                  target_lang: str = "zh") -> str:
        """
        翻译文本

        Args:
            text: 待翻译文本
            source_lang: 源语言代码 ("auto" 自动检测)
            target_lang: 目标语言代码

        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return ""

        # 处理缓存
        cache_key = (text.strip(), source_lang, target_lang)
        if cache_key in self._cache:
            return self._cache[cache_key]

        src = self.LANG_MAP.get(source_lang, source_lang)
        tgt = self.LANG_MAP.get(target_lang, target_lang)

        # MyMemory 不支持 auto，用 e=URL 编码猜测
        # 实际上 MyMemory 的 langpair 需要明确指定源语言
        # 如果是 auto，尝试用文本特征判断
        if src == "Autodetect":
            src = self._detect_language(text)

        result = self._translate_mymemory(text.strip(), src, tgt)

        if result:
            self._cache[cache_key] = result
            # 缓存过大时清空
            if len(self._cache) > 500:
                self._cache.clear()

        return result if result else text.strip()

    def _translate_mymemory(self, text: str, source: str, target: str) -> str:
        """使用 MyMemory API 翻译"""
        langpair = f"{source}|{target}"

        # 限流
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

        try:
            url = "https://api.mymemory.translated.net/get"
            params = {
                "q": text,
                "langpair": langpair,
            }
            r = self._session.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                translated = data.get("responseData", {}).get("translatedText", "")
                if translated and "MYMEMORY WARNING" not in translated.upper():
                    return translated
                # 尝试从 matches 中获取
                matches = data.get("matches", [])
                if matches:
                    return matches[0].get("translation", "")
            else:
                print(f"[Translator] MyMemory HTTP {r.status_code}")
        except Exception as e:
            print(f"[Translator] MyMemory 请求失败: {e}")

        return ""

    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""
        # 检查中文字符
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return "zh-CN"

        # 检查西里尔字符 (俄语)
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        if cyrillic_chars > len(text) * 0.3:
            return "ru"

        # 默认英语
        return "en"
