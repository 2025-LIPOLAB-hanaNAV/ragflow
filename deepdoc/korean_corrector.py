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
from typing import Optional, List


class KoreanTextCorrector:
    """Korean text correction service using Ollama gemma3:12b model"""

    def __init__(self, ollama_url: str = "http://host.docker.internal:11435", model: str = "gemma3:12b"):
        self.ollama_url = ollama_url
        self.model = model
        self.correction_prompt = """다음 한국어 텍스트의 맞춤법과 띄어쓰기를 자연스럽게 교정해주세요.
OCR로 인식된 텍스트이므로 문맥을 고려하여 올바른 형태로 수정해주세요.
원본 의미를 최대한 보존하면서 자연스러운 한국어로 만들어주세요.

원본 텍스트: {text}

교정된 텍스트:"""

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

    def clean_response(self, response_text: str) -> str:
        """Clean and extract the corrected text from LLM response"""
        # Remove any markdown formatting
        response_text = re.sub(r'```[^`]*```', '', response_text)
        response_text = re.sub(r'`[^`]*`', '', response_text)

        # Split by lines and find the most relevant line
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        if not lines:
            return response_text.strip()

        # Look for lines that don't contain common prompt markers
        filtered_lines = []
        for line in lines:
            if not any(marker in line.lower() for marker in [
                '교정된', '원본', '텍스트:', '다음', '맞춤법', '띄어쓰기', 'ocr'
            ]):
                filtered_lines.append(line)

        # Return the longest meaningful line
        if filtered_lines:
            return max(filtered_lines, key=len)
        else:
            return lines[-1] if lines else response_text.strip()

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
                    cleaned_text = self.clean_response(corrected_text)
                    # Basic validation: corrected text shouldn't be dramatically different in length
                    if len(cleaned_text) > 0 and len(cleaned_text) < len(text) * 3:
                        return cleaned_text
                    else:
                        logging.warning(f"Korean correction produced unexpected result length. Original: {len(text)}, Corrected: {len(cleaned_text)}")
                        return text
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