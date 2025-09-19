#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
import logging
import re
import requests
from difflib import SequenceMatcher
from typing import List


class KoreanTextCorrector:
    """Korean text correction service using Ollama gemma3:12b model"""

    def __init__(self, ollama_url: str = "http://host.docker.internal:11435", model: str = "gemma3:12b"):
        self.ollama_url = ollama_url
        self.model = model
        self.correction_prompt = """다음 한국어 텍스트의 맞춤법과 띄어쓰기를 자연스럽게 교정해주세요.
OCR로 인식된 텍스트이므로 문맥을 고려하되, 아래 제약을 반드시 지켜주세요.
- 요약하거나 문장을 삭제·추가·재배열하지 않습니다.
- 줄바꿈, 문장 수, 문단 수를 가능한 한 그대로 유지합니다.
- 원본에 없는 설명이나 주석을 덧붙이지 않습니다.
- 답변에는 교정된 문장만 포함하고 다른 말은 하지 않습니다.

<원본>
{text}
</원본>

<교정본>"""

    def is_korean_text(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        if not text:
            return False

        korean_chars = 0
        total_chars = 0

        for char in text:
            if char.strip():
                total_chars += 1
                code = ord(char)
                if (0xAC00 <= code <= 0xD7AF) or (0x1100 <= code <= 0x11FF) or (0x3130 <= code <= 0x318F):
                    korean_chars += 1

        if total_chars == 0:
            return False

        # If more than 10% of characters are Korean, consider it Korean text
        return (korean_chars / total_chars) > 0.1

    def _strip_fences(self, text: str) -> str:
        """Remove common fence markers without altering order."""
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"```", "", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        return text

    def clean_response(self, response_text: str, original_text: str) -> str:
        """Clean LLM output while keeping structure and fall back if invalid."""
        cleaned = self._strip_fences(response_text or "").strip()

        cleaned = re.sub(r"^\s*<교정본>\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</교정본>\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(교정된\s*텍스트\s*[:：])\s*", "", cleaned, flags=re.IGNORECASE)

        if not cleaned:
            return original_text

        if not self._is_valid_correction(original_text, cleaned):
            logging.warning(
                "Korean correction rejected due to large deviation. original_len=%s corrected_len=%s",
                len(original_text.strip()),
                len(cleaned.strip()),
            )
            return original_text

        return cleaned

    @staticmethod
    def _is_valid_correction(original: str, corrected: str) -> bool:
        """Ensure corrected text stays close to the original content."""
        if not original.strip():
            return True

        orig_text = original.strip()
        corr_text = corrected.strip()
        orig_len = len(orig_text)
        corr_len = len(corr_text)

        if corr_len == 0:
            return False

        if orig_len <= 30:
            # Very short snippets are accepted as long as they are non-empty
            return True

        ratio = corr_len / orig_len if orig_len else 1.0
        if ratio < 0.55 or ratio > 1.45:
            return False

        orig_lines = [l for l in orig_text.splitlines() if l.strip()]
        corr_lines = [l for l in corr_text.splitlines() if l.strip()]
        if orig_lines:
            line_ratio = len(corr_lines) / len(orig_lines) if len(orig_lines) else 1.0
            if line_ratio < 0.5 or line_ratio > 1.7:
                return False

        orig_compact = re.sub(r"\s+", "", orig_text)
        corr_compact = re.sub(r"\s+", "", corr_text)
        if not corr_compact:
            return False

        similarity = SequenceMatcher(None, orig_compact, corr_compact).ratio()
        if similarity < 0.6:
            return False

        return True

    def correct_text(self, text: str) -> str:
        """Correct Korean text using Ollama"""
        if not text or not text.strip():
            return text

        # Skip correction if text is too short or doesn't contain Korean
        if len(text.strip()) < 3 or not self.is_korean_text(text):
            return text

        try:
            prompt = self.correction_prompt.format(text=text)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 512
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                corrected_text = result.get('response', '').strip()

                if corrected_text:
                    cleaned_text = self.clean_response(corrected_text, text)
                    return cleaned_text
                else:
                    logging.warning("Empty response from Korean corrector")
                    return text
            else:
                logging.error(f"Korean correction API error: {response.status_code}")
                return text

        except Exception as e:
            logging.error(f"Korean text correction failed: {str(e)}")
            return text

    def correct_text_batch(self, texts: List[str]) -> List[str]:
        """Correct multiple texts in batch"""
        corrected_texts = []
        for text in texts:
            corrected_texts.append(self.correct_text(text))
        return corrected_texts


# Global instance
_korean_corrector = None


def get_korean_corrector() -> KoreanTextCorrector:
    """Get or create Korean corrector instance"""
    global _korean_corrector
    if _korean_corrector is None:
        _korean_corrector = KoreanTextCorrector()
    return _korean_corrector


def correct_korean_text(text: str) -> str:
    """Convenience function to correct Korean text"""
    corrector = get_korean_corrector()
    return corrector.correct_text(text)
