"""
Ollama LLM Client
Direct integration with local Ollama server for fine-tuned models
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Direct Ollama client for local LLM interactions"""

    def __init__(self, model_name: str = "nfl_playbypay_lfm2:latest"):
        """
        Initialize Ollama client

        Args:
            model_name: Ollama model name to use
        """
        self.model_name = model_name
        self.timeout = 30  # seconds

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Get completion from Ollama

        Args:
            prompt: User prompt
            model: Model to use (default: nfl_playbypay_lfm2:latest)
            temperature: Sampling temperature (not used by Ollama CLI)
            max_tokens: Maximum tokens to generate (not used by Ollama CLI)
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        # Use specified model or default
        model_name = model or self.model_name

        try:
            # Run Ollama command
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "ollama",
                    "run",
                    model_name,
                    full_prompt,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self.timeout,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(
                    f"Ollama error (exit code {process.returncode}): {error_msg}"
                )
                raise Exception(f"Ollama error: {error_msg}")

            response = stdout.decode().strip()
            if not response:
                raise Exception("Empty response from Ollama")

            return response

        except asyncio.TimeoutError:
            raise Exception(f"Ollama call timed out after {self.timeout} seconds")
        except FileNotFoundError:
            raise Exception(
                "Ollama not found. Please install Ollama: https://ollama.ai"
            )
        except Exception as e:
            logger.error(f"Failed to call Ollama: {str(e)}")
            raise

    async def complete_json(
        self, prompt: str, model: Optional[str] = None, temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Get JSON response from Ollama

        Args:
            prompt: User prompt (should request JSON format)
            model: Model to use
            temperature: Lower temperature for structured output (not used by Ollama CLI)

        Returns:
            Parsed JSON response
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only. Do not include any other text or formatting."

        response = await self.complete(
            prompt=json_prompt, model=model, temperature=temperature
        )

        # Try to extract JSON from response
        try:
            # Clean the response
            response = response.strip()

            # Handle potential markdown formatting
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            return {"error": "Failed to parse JSON", "raw": response}

    async def analyze_market(
        self,
        market_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None,
        game_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze market opportunity using LLM

        Args:
            market_data: Neural market data
            sentiment_data: Twitter sentiment data
            game_data: ESPN game data

        Returns:
            Analysis with trading recommendation
        """
        prompt = f"""
        Analyze this trading opportunity:
        
        Market Data: {market_data}
        Sentiment Data: {sentiment_data or 'N/A'}
        Game Data: {game_data or 'N/A'}
        
        Provide analysis as JSON with:
        - signal: BUY_YES, BUY_NO, or HOLD
        - confidence: 0-1 score
        - reasoning: brief explanation
        - risk_level: low, medium, or high
        - suggested_size: percentage of capital (0-25)
        """

        return await self.complete_json(prompt, temperature=0.2)


# Singleton instance for easy import
llm_client = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton"""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client
