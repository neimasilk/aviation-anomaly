"""
DeepSeek API client for data augmentation and analysis.

Uses DeepSeek API for:
- Synthetic data generation
- Data augmentation
- Text analysis
"""
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import config


class DeepSeekClient:
    """Client for DeepSeek API (OpenAI-compatible)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key. If None, loads from config.
            base_url: API base URL. If None, uses default DeepSeek URL.
        """
        self.api_key = api_key or config.deepseek_api_key
        self.base_url = base_url or config.deepseek_base_url

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using DeepSeek API.

        Args:
            prompt: Input prompt
            model: Model name (deepseek-chat, deepseek-coder)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content

    def generate_cvr_augmentation(
        self,
        original_utterance: str,
        context: Optional[str] = None,
        variations: int = 3,
        stress_level: str = "normal"
    ) -> List[str]:
        """
        Generate augmented CVR utterances.

        Args:
            original_utterance: Original pilot utterance
            context: Previous utterances for context
            variations: Number of variations to generate
            stress_level: stress level (normal, elevated, critical)

        Returns:
            List of augmented utterances
        """
        prompt = f"""You are assisting with aviation safety research. Generate {variations} variations
of the following pilot cockpit utterance while maintaining the same meaning and aviation context.

Original utterance: "{original_utterance}"
Stress level: {stress_level}"""

        if context:
            prompt += f"\nContext (previous utterances):\n{context}"

        prompt += """

Generate only the variations, one per line, without numbering or additional text.
Maintain the aviation terminology and communication style."""

        response = self.generate(prompt, temperature=0.8)
        variations_list = [line.strip() for line in response.split("\n") if line.strip()]
        return variations_list[:variations]

    def generate_synthetic_cvr_conversation(
        self,
        scenario: str,
        num_utterances: int = 10,
        stress_progression: str = "normal_to_critical"
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic CVR conversation for training data.

        Args:
            scenario: Aviation scenario description
            num_utterances: Number of utterances to generate
            stress_progression: How stress should progress

        Returns:
            List of utterances with speaker and metadata
        """
        prompt = f"""Generate a realistic cockpit voice recorder conversation for aviation safety research.

Scenario: {scenario}
Number of utterances: {num_utterances}
Stress progression: {stress_progression}

Requirements:
- Include Captain (CAP) and First Officer (FO) roles
- Use realistic aviation phraseology
- Progress from normal communication to the specified stress level
- Include timestamps in format [HH:MM:SS]

Output format (JSON-like):
[
    {{"timestamp": "[14:25:30]", "speaker": "CAP", "utterance": "...", "stress_level": "normal"}},
    ...
]"""

        response = self.generate(prompt, temperature=0.9, model="deepseek-chat")
        # Parse response - in production, add proper JSON parsing
        return response

    def analyze_utterance_stress(
        self,
        utterance: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze stress level in an utterance.

        Args:
            utterance: Pilot utterance to analyze
            context: Previous context

        Returns:
            Analysis with stress level and indicators
        """
        prompt = f"""Analyze the following cockpit utterance for stress indicators.

Utterance: "{utterance}"
{"Context: " + context if context else ""}

Provide analysis in JSON format:
{{
    "stress_level": "normal|elevated|critical",
    "indicators": ["list", "of", "indicators"],
    "confidence": 0.0-1.0
}}"""

        response = self.generate(prompt, temperature=0.3)
        # In production, add proper JSON parsing
        return {"raw_analysis": response}

    def augment_dataset(
        self,
        utterances: List[str],
        augmentations_per_sample: int = 2
    ) -> List[str]:
        """
        Augment a dataset of utterances.

        Args:
            utterances: List of original utterances
            augmentations_per_sample: Number of variations per sample

        Returns:
            Augmented dataset
        """
        augmented = []
        for utterance in utterances:
            augmented.append(utterance)  # Keep original
            try:
                variations = self.generate_cvr_augmentation(
                    utterance,
                    variations=augmentations_per_sample
                )
                augmented.extend(variations)
            except Exception as e:
                print(f"Error augmenting '{utterance[:30]}...': {e}")
        return augmented


# Singleton instance
_deepseek_client: Optional[DeepSeekClient] = None


def get_deepseek_client() -> DeepSeekClient:
    """Get or create DeepSeek client singleton."""
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = DeepSeekClient()
    return _deepseek_client


if __name__ == "__main__":
    # Test the client
    client = get_deepseek_client()

    # Test basic generation
    print("Testing basic generation:")
    print(client.generate("Say hello in the style of a pilot"))

    # Test CVR augmentation
    print("\nTesting CVR augmentation:")
    augmentations = client.generate_cvr_augmentation(
        "We're losing altitude, need to increase power."
    )
    for i, aug in enumerate(augmentations, 1):
        print(f"{i}. {aug}")
