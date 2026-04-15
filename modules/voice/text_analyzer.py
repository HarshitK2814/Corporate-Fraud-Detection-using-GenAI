
import os
from groq import Groq
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

class TextAnalyzer:
    """
    Analyzes text transcripts for Semantic Drift and Deception.
    Uses Groq (Llama 3 70B) for fast inference.
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        
        self.client = Groq(api_key=self.api_key)

    def analyze_semantics(self, text):
        """
        Detects vagueness, evasion, and semantic drift in earnings call text.
        Returns a score (0.0 to 1.0) and analysis.
        """
        if not text or len(text.strip()) < 10:
            return {"error": "Text too short for analysis", "score": 0.0}

        prompt = f"""
        Analyze the following excerpt from a corporate earnings call for signs of deception, specifically "Semantic Drift".
        
        Semantic Drift Definition:
        - Moving from specific metrics to vague qualitative statements.
        - Answering a direct question with an unrelated answer (evasion).
        - Excessive use of "we believe", "hope", "optimistic", "future outlook" without data.
        - Complex, convoluted sentence structures to hide bad news.

        Excerpt:
        "{text[:4000]}"  # Truncate to avoid context limit if huge

        Task:
        1. Identify specific vague or evasive phrases.
        2. Assign a "Deception Risk Score" from 0.0 (Clear/Honest) to 1.0 (Highly Evasive).
        3. Explain WHY.

        Output JSON format only:
        {{
            "risk_score": 0.0 to 1.0,
            "verdict": "Clear / Mild Evasion / Highly Evasive",
            "vagueness_check": "Detailed explanation of vague phrases found...",
            "evasion_check": "Did they dodge questions? explain...",
            "key_phrases": ["phrase 1", "phrase 2"]
        }}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a forensic linguistic analyst specializing in financial deception detection. Create valid JSON output."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                response_format={"type": "json_object"} # Force JSON
            )
            
            result = chat_completion.choices[0].message.content
            # Ensure it's parsed as dict
            import json
            return json.loads(result)

        except Exception as e:
            print(f"Groq API Error: {e}")
            return {
                "risk_score": 0.0,
                "error": str(e)
            }

if __name__ == "__main__":
    ta = TextAnalyzer()
    # sample = "We are optimistic about the future despite the headwinds. We believe that our long term strategy will pay off eventually."
    # print(ta.analyze_semantics(sample))
