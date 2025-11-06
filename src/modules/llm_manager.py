# Create new file: /src/modules/llm_manager.py

import os
import logging

class LLMManager:
    """A centralized manager for initializing and accessing LLM clients."""
    
    def __init__(self):
        self.clients = {}
        self.available_llms = []
        self._setup_clients()

    def _setup_clients(self):
        """Initializes all configured LLM clients."""
        # --- Existing Clients ---
        self._setup_groq()
        self._setup_gemini()
        self._setup_cohere()
        
        # --- New Clients (Step 2) ---
        # We will add DeepSeek and Mistral here in the next task.
        
        if not self.clients:
            logging.error("❌ CRITICAL: No LLM clients could be initialized.")
        else:
            self.available_llms = list(self.clients.keys())
            logging.info(f"✅ LLM Manager initialized. Available clients: {self.available_llms}")

    def get_client(self, provider_name):
        """Returns the client for a given provider."""
        return self.clients.get(provider_name)

    def _setup_groq(self):
        if os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                self.clients['groq'] = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logging.info("✅ LLM Manager: Groq client loaded.")
            except ImportError:
                logging.warning("⚠️ LLM Manager: 'groq' library not installed. Skipping Groq.")
            except Exception as e:
                logging.error(f"❌ LLM Manager: Groq initialization failed: {e}")

    def _setup_gemini(self):
        if os.getenv("GEMINI_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                # Using a slightly different model name to avoid conflicts if needed later
                self.clients['gemini'] = genai.GenerativeModel('gemini-1.5-flash')
                logging.info("✅ LLM Manager: Gemini client loaded.")
            except ImportError:
                logging.warning("⚠️ LLM Manager: 'google-generativeai' library not installed. Skipping Gemini.")
            except Exception as e:
                logging.error(f"❌ LLM Manager: Gemini initialization failed: {e}")

    def _setup_cohere(self):
        if os.getenv("COHERE_API_KEY"):
            try:
                import cohere
                self.clients['cohere'] = cohere.Client(os.getenv("COHERE_API_KEY"))
                logging.info("✅ LLM Manager: Cohere client loaded.")
            except ImportError:
                logging.warning("⚠️ LLM Manager: 'cohere' library not installed. Skipping Cohere.")
            except Exception as e:
                logging.error(f"❌ LLM Manager: Cohere initialization failed: {e}")

# This makes the manager a singleton, so it's initialized only once.
llm_manager = LLMManager()
