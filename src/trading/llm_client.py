"""
OpenRouter LLM Client
Direct API integration without Agno wrapper
"""

import os
import aiohttp
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LLMClient:
    """Direct OpenRouter API client for LLM interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenRouter API key (or from environment)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = "google/gemini-2.5-flash"
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Get completion from OpenRouter API
        
        Args:
            prompt: User prompt
            model: Model to use (default: gemini-2.5-flash)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/kalshi-agent",
                    "X-Title": "Kalshi Trading Agent"
                },
                json={
                    "model": model or self.default_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"OpenRouter API error: {error}")
                    raise Exception(f"API error: {resp.status}")
                
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
    
    async def complete_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Get JSON response from LLM
        
        Args:
            prompt: User prompt (should request JSON format)
            model: Model to use
            temperature: Lower temperature for structured output
            
        Returns:
            Parsed JSON response
        """
        import json
        
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only."
        
        response = await self.complete(
            prompt=json_prompt,
            model=model,
            temperature=temperature
        )
        
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response}")
            return {"error": "Failed to parse JSON", "raw": response}
    
    async def analyze_market(
        self,
        market_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None,
        game_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze market opportunity using LLM
        
        Args:
            market_data: Kalshi market data
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