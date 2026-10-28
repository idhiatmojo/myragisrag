"""
Modul Multi-Model LLM Provider (Google Gemini, xAI Grok, Hugging Face, OpenAI, Ollama, DeepSeek, dll.)
Menyediakan antarmuka terpadu (Unified Interface) untuk ekstraksi terstruktur (Structured Output)
dan sintesis laporan akademis S2 AI dengan penanganan fallback yang tangguh.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Type, TypeVar
from pydantic import BaseModel

from src.config import (
    LLM_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_LLM_MODEL,
    GROK_API_KEY,
    GROK_LLM_MODEL,
    GROK_BASE_URL,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_LLM_MODEL,
    HUGGINGFACE_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_LLM_MODEL,
    OPENAI_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_LLM_MODEL,
    DEEPSEEK_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_LLM_MODEL,
    OPENROUTER_BASE_URL,
    CUSTOM_LLM_API_KEY,
    CUSTOM_LLM_MODEL,
    CUSTOM_LLM_BASE_URL,
)

T = TypeVar("T", bound=BaseModel)


def extract_json_from_response(raw_text: str) -> Dict[str, Any]:
    """
    Mengekstrak objek JSON secara tangguh dari respons model teks.
    Mendukung format Markdown ```json ... ```, teks pengantar/penutup,
    dan sanitasi trailing commas.
    """
    if not raw_text or not raw_text.strip():
        return {}

    text = raw_text.strip()

    # 1. Coba langsung json.loads
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Hapus markdown code fences
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, text, re.IGNORECASE)
    if match:
        fence_content = match.group(1).strip()
        try:
            return json.loads(fence_content)
        except Exception:
            text = fence_content

    # 3. Cari pasangan kurung kurawal terluar { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = text[first_brace : last_brace + 1].strip()
        try:
            return json.loads(snippet)
        except Exception:
            # 4. Bersihkan trailing commas jika ada (e.g., {"a": 1, })
            cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                return json.loads(cleaned)
            except Exception:
                pass

    return {}


class BaseLLMClient(ABC):
    """Kelas dasar abstrak untuk semua penyedia model LLM."""

    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name
        self.is_configured: bool = False
        self.status_message: str = ""

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema_cls: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[Optional[T], Dict[str, int]]:
        """
        Menghasilkan output terstruktur yang divalidasi ke dalam schema Pydantic.
        Mengembalikan tuple: (parsed_instance, token_usage_dict)
        """
        pass


class GeminiLLMClient(BaseLLMClient):
    """Penyedia model Google Gemini resmi via library google-genai."""

    def __init__(self, api_key: str, model_name: str):
        super().__init__(provider_name="gemini", model_name=model_name)
        self.api_key = api_key
        self._client = None

        if not self.api_key or self.api_key == "masukkan_api_key_gemini_anda_disini":
            self.status_message = "GEMINI_API_KEY belum disetel di file .env."
            self.is_configured = False
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self.is_configured = True
            self.status_message = f"Google Gemini API aktif ({self.model_name})"
        except Exception as exc:
            self.status_message = f"Gagal inisialisasi Gemini Client: {exc}"
            self.is_configured = False

    def generate_structured(
        self,
        prompt: str,
        schema_cls: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[Optional[T], Dict[str, int]]:
        if not self.is_configured or self._client is None:
            return None, {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}

        tokens = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Percobaan 1: Native Schema Enforcement
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_cls,
                temperature=temperature,
            )
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=config,
            )
            meta = getattr(response, "usage_metadata", None)
            if meta:
                p_tok = getattr(meta, "prompt_token_count", 0) or 0
                c_tok = getattr(meta, "candidates_token_count", 0) or 0
                t_tok = getattr(meta, "total_token_count", 0) or (p_tok + c_tok)
                tokens = {"prompt_tokens": p_tok, "candidate_tokens": c_tok, "total_tokens": t_tok}

            data = json.loads(response.text)
            return schema_cls(**data), tokens
        except Exception as exc:
            # Percobaan 2: Fallback teks + Regex JSON extractor
            try:
                schema_json_hint = json.dumps(schema_cls.model_json_schema().get("properties", {}), indent=2)
                fallback_prompt = (
                    f"{full_prompt}\n\n"
                    f"PENTING: Keluarkan HANYA JSON murni yang sesuai dengan skema properti berikut:\n"
                    f"{schema_json_hint}\n"
                )
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=fallback_prompt,
                )
                data = extract_json_from_response(response.text)
                if data:
                    return schema_cls(**data), tokens
            except Exception as exc2:
                print(f"[WARN] Gemini ekstraksi gagal: {exc} | {exc2}")

        return None, tokens


class OpenAICompatibleLLMClient(BaseLLMClient):
    """
    Penyedia model universal untuk antarmuka OpenAI-Compatible:
    xAI Grok, Hugging Face Router, OpenAI, Ollama, DeepSeek, OpenRouter, dll.
    """

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        model_name: str,
        base_url: str,
        default_key_placeholder: str = "",
    ):
        super().__init__(provider_name=provider_name, model_name=model_name)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

        # Khusus Ollama lokal, tidak memerlukan API key sungguhan
        if provider_name == "ollama":
            self.api_key = self.api_key or "ollama"

        if not self.api_key or self.api_key == default_key_placeholder:
            self.status_message = f"API Key untuk {provider_name.upper()} belum disetel di .env."
            self.is_configured = False
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.is_configured = True
            self.status_message = (
                f"{provider_name.upper()} API aktif ({self.model_name} @ {self.base_url})"
            )
        except Exception as exc:
            self.status_message = f"Gagal inisialisasi client {provider_name}: {exc}"
            self.is_configured = False

    def generate_structured(
        self,
        prompt: str,
        schema_cls: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Tuple[Optional[T], Dict[str, int]]:
        if not self.is_configured or self._client is None:
            return None, {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}

        tokens = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}
        schema_props = schema_cls.model_json_schema().get("properties", {})
        schema_instruction = (
            f"Anda WAJIB memberikan jawaban HANYA dalam format JSON valid sesuai kunci-kunci berikut:\n"
            f"{json.dumps(schema_props, indent=2, ensure_ascii=False)}\n"
            f"Jangan sertakan teks pengantar atau markdown selain JSON itu sendiri."
        )

        sys_msg = (
            f"{system_prompt}\n\n{schema_instruction}"
            if system_prompt
            else schema_instruction
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        # 1. Coba Structured Outputs (client.beta.chat.completions.parse)
        try:
            completion = self._client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=schema_cls,
                temperature=temperature,
            )
            usage = getattr(completion, "usage", None)
            if usage:
                p_tok = getattr(usage, "prompt_tokens", 0) or 0
                c_tok = getattr(usage, "completion_tokens", 0) or 0
                t_tok = getattr(usage, "total_tokens", 0) or (p_tok + c_tok)
                tokens = {"prompt_tokens": p_tok, "candidate_tokens": c_tok, "total_tokens": t_tok}

            if completion.choices and completion.choices[0].message.parsed:
                return completion.choices[0].message.parsed, tokens
        except Exception:
            # Banyak penyedia seperti Grok, HF Router, atau Ollama belum mendukung beta.parse
            pass

        # 2. Coba Chat Completions dengan JSON Object mode
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            usage = getattr(completion, "usage", None)
            if usage:
                p_tok = getattr(usage, "prompt_tokens", 0) or 0
                c_tok = getattr(usage, "completion_tokens", 0) or 0
                t_tok = getattr(usage, "total_tokens", 0) or (p_tok + c_tok)
                tokens = {"prompt_tokens": p_tok, "candidate_tokens": c_tok, "total_tokens": t_tok}

            raw_content = completion.choices[0].message.content or ""
            data = extract_json_from_response(raw_content)
            if data:
                return schema_cls(**data), tokens
        except Exception:
            # 3. Fallback tanpa response_format parameter jika penyedia menolaknya
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                usage = getattr(completion, "usage", None)
                if usage:
                    p_tok = getattr(usage, "prompt_tokens", 0) or 0
                    c_tok = getattr(usage, "completion_tokens", 0) or 0
                    t_tok = getattr(usage, "total_tokens", 0) or (p_tok + c_tok)
                    tokens = {"prompt_tokens": p_tok, "candidate_tokens": c_tok, "total_tokens": t_tok}

                raw_content = completion.choices[0].message.content or ""
                data = extract_json_from_response(raw_content)
                if data:
                    return schema_cls(**data), tokens
            except Exception as exc:
                print(f"[WARN] Panggilan API ke {self.provider_name} ({self.model_name}) gagal: {exc}")

        return None, tokens


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMClient:
    """
    Factory method untuk mendapatkan instance LLM Client sesuai pengaturan .env
    atau argumen CLI.
    """
    chosen_provider = (provider or LLM_PROVIDER or "gemini").strip().lower()

    # Alias normalisasi
    if chosen_provider in ("gemini", "google"):
        chosen_model = model or GEMINI_LLM_MODEL
        client = GeminiLLMClient(api_key=GEMINI_API_KEY, model_name=chosen_model)

    elif chosen_provider in ("grok", "xai"):
        chosen_model = model or GROK_LLM_MODEL
        client = OpenAICompatibleLLMClient(
            provider_name="grok",
            api_key=GROK_API_KEY,
            model_name=chosen_model,
            base_url=GROK_BASE_URL,
            default_key_placeholder="masukkan_api_key_grok_anda_disini",
        )

    elif chosen_provider in ("huggingface", "hf", "hugface"):
        chosen_model = model or HUGGINGFACE_LLM_MODEL
        client = OpenAICompatibleLLMClient(
            provider_name="huggingface",
            api_key=HUGGINGFACE_API_KEY,
            model_name=chosen_model,
            base_url=HUGGINGFACE_BASE_URL,
            default_key_placeholder="masukkan_token_hf_anda_disini",
        )

    elif chosen_provider == "openai":
        chosen_model = model or OPENAI_LLM_MODEL
        client = OpenAICompatibleLLMClient(
            provider_name="openai",
            api_key=OPENAI_API_KEY,
            model_name=chosen_model,
            base_url=OPENAI_BASE_URL,
            default_key_placeholder="masukkan_api_key_openai_anda_disini",
        )

    elif chosen_provider in ("ollama", "local"):
        chosen_model = model or OLLAMA_LLM_MODEL
        client = OpenAICompatibleLLMClient(
            provider_name="ollama",
            api_key="ollama",
            model_name=chosen_model,
            base_url=OLLAMA_BASE_URL,
        )

    elif chosen_provider == "deepseek":
        chosen_model = model or DEEPSEEK_LLM_MODEL
        client = OpenAICompatibleLLMClient(
            provider_name="deepseek",
            api_key=DEEPSEEK_API_KEY,
            model_name=chosen_model,
            base_url=DEEPSEEK_BASE_URL,
            default_key_placeholder="masukkan_api_key_deepseek_anda_disini",
        )

    elif chosen_provider == "openrouter":
        chosen_model = model or OPENROUTER_LLM_MODEL
        client = OpenAICompatibleLLMClient(
            provider_name="openrouter",
            api_key=OPENROUTER_API_KEY,
            model_name=chosen_model,
            base_url=OPENROUTER_BASE_URL,
            default_key_placeholder="masukkan_api_key_openrouter_anda_disini",
        )

    elif chosen_provider == "custom":
        chosen_model = model or CUSTOM_LLM_MODEL
        client = OpenAICompatibleLLMClient(
            provider_name="custom",
            api_key=CUSTOM_LLM_API_KEY,
            model_name=chosen_model,
            base_url=CUSTOM_LLM_BASE_URL,
        )

    else:
        # Fallback default ke Gemini
        chosen_model = model or GEMINI_LLM_MODEL
        client = GeminiLLMClient(api_key=GEMINI_API_KEY, model_name=chosen_model)

    return client

