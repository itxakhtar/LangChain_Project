import streamlit as st
import os
import asyncio
import time
import logging
import traceback
import re
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

# ----------------------------------------------------------------------
# Logging & Environment
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(
    page_title="Multi-Model AI Arena",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Custom CSS (unchanged, kept for professional look)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .notification-card {
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .notification-error { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-left: 5px solid #f56565; color: white; }
    .notification-warning { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-left: 5px solid #ed8936; color: white; }
    .notification-info { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-left: 5px solid #4299e1; color: white; }
    .notification-success { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-left: 5px solid #48bb78; color: white; }
    .notification-content { display: flex; align-items: center; gap: 15px; }
    .notification-icon { font-size: 32px; }
    .notification-message { flex: 1; }
    .notification-title { font-weight: bold; font-size: 18px; margin-bottom: 5px; }
    .notification-description { font-size: 14px; opacity: 0.95; }
    .model-card { background: white; border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s; }
    .model-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
    .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
    .status-success { background-color: #c6f6d5; color: #22543d; }
    .status-error { background-color: #fed7d7; color: #742a2a; }
    .custom-spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,.3); border-radius: 50%; border-top-color: white; animation: spin 1s ease-in-out infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .main-header { text-align: center; font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Error handling classes (unchanged, but fixed rate-limit parsing)
# ----------------------------------------------------------------------
class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    QUOTA_EXCEEDED = "quota_exceeded"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"

@dataclass
class ErrorDetails:
    error_type: ErrorType
    user_message: str
    technical_details: str
    retry_after: Optional[int] = None
    next_available_time: Optional[datetime] = None

class ErrorHandler:
    ERROR_MESSAGES = {
        ErrorType.RATE_LIMIT: {"title": "Rate Limit Reached", "message": "You've reached the rate limit for this model.", "action": "Please wait a moment before trying again."},
        ErrorType.AUTHENTICATION: {"title": "Authentication Failed", "message": "Unable to authenticate with the API.", "action": "Please check your API key and try again."},
        ErrorType.TIMEOUT: {"title": "Request Timeout", "message": "The request took too long to complete.", "action": "The model is currently busy. Please try again."},
        ErrorType.CONNECTION: {"title": "Connection Error", "message": "Unable to connect to the service.", "action": "Please check your internet connection and try again."},
        ErrorType.QUOTA_EXCEEDED: {"title": "Quota Exceeded", "message": "You've exceeded your API quota.", "action": "Please check your usage limits or upgrade your plan."},
        ErrorType.MAINTENANCE: {"title": "Under Maintenance", "message": "This model is currently undergoing maintenance.", "action": "Please try again later or use another model."},
        ErrorType.UNAVAILABLE: {"title": "Temporarily Unavailable", "message": "The service is currently unavailable.", "action": "Please try again in a few minutes."},
        ErrorType.UNKNOWN: {"title": "Unexpected Error", "message": "An unexpected error occurred.", "action": "Please try again or contact support if the issue persists."}
    }

    @staticmethod
    def classify_error(error: Exception) -> ErrorDetails:
        error_str = str(error).lower()
        if any(k in error_str for k in ['rate limit', 'too many requests', 'rate_limit']):
            return ErrorHandler._create_rate_limit_error(error)
        elif any(k in error_str for k in ['authentication', 'api key', 'unauthorized', 'invalid key']):
            return ErrorDetails(ErrorType.AUTHENTICATION, ErrorHandler._format_user_message(ErrorType.AUTHENTICATION), str(error))
        elif any(k in error_str for k in ['timeout', 'timed out']):
            return ErrorDetails(ErrorType.TIMEOUT, ErrorHandler._format_user_message(ErrorType.TIMEOUT), str(error))
        elif any(k in error_str for k in ['connection', 'network', 'socket']):
            return ErrorDetails(ErrorType.CONNECTION, ErrorHandler._format_user_message(ErrorType.CONNECTION), str(error))
        elif any(k in error_str for k in ['quota', 'exceeded', 'limit reached']):
            return ErrorDetails(ErrorType.QUOTA_EXCEEDED, ErrorHandler._format_user_message(ErrorType.QUOTA_EXCEEDED), str(error))
        else:
            return ErrorDetails(ErrorType.UNKNOWN, ErrorHandler._format_user_message(ErrorType.UNKNOWN), str(error))

    @staticmethod
    def _create_rate_limit_error(error: Exception) -> ErrorDetails:
        error_str = str(error).lower()
        retry_after = None
        match = re.search(r'retry after (\d+)', error_str)
        if match:
            retry_after = int(match.group(1))
        next_time = datetime.now() + timedelta(seconds=retry_after) if retry_after else None
        return ErrorDetails(
            ErrorType.RATE_LIMIT,
            ErrorHandler._format_rate_limit_message(retry_after, next_time),
            str(error),
            retry_after,
            next_time
        )

    @staticmethod
    def _format_user_message(error_type: ErrorType) -> str:
        m = ErrorHandler.ERROR_MESSAGES[error_type]
        return f"**{m['title']}**\n\n{m['message']}\n\n{m['action']}"

    @staticmethod
    def _format_rate_limit_message(retry_after: Optional[int], next_time: Optional[datetime]) -> str:
        if next_time:
            return f"**Rate Limit Reached**\n\nPlease try again at {next_time.strftime('%I:%M %p')}."
        elif retry_after:
            return f"**Rate Limit Reached**\n\nPlease wait {retry_after} seconds."
        return "**Rate Limit Reached**\n\nPlease try again in a few minutes."

class RetryManager:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute_with_retry(self, func: Callable, *args, progress_callback: Optional[Callable] = None, **kwargs):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if progress_callback and attempt > 0:
                    progress_callback(f"Retry attempt {attempt}/{self.max_retries}...")
                result = await func(*args, **kwargs)
                if progress_callback:
                    progress_callback("Success!")
                return result
            except Exception as e:
                last_error = e
                error_details = ErrorHandler.classify_error(e)
                if error_details.error_type in [ErrorType.AUTHENTICATION, ErrorType.QUOTA_EXCEEDED]:
                    raise
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    if progress_callback:
                        progress_callback(f"Error. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise last_error

class NotificationComponent:
    @staticmethod
    def show_error(error_details: ErrorDetails, model_name: str = None):
        icon_map = {ErrorType.RATE_LIMIT: "⏰", ErrorType.AUTHENTICATION: "🔑", ErrorType.TIMEOUT: "⏱️",
                    ErrorType.CONNECTION: "🌐", ErrorType.QUOTA_EXCEEDED: "📊", ErrorType.MAINTENANCE: "🔧",
                    ErrorType.UNAVAILABLE: "⚠️", ErrorType.UNKNOWN: "❓"}
        icon = icon_map.get(error_details.error_type, "⚠️")
        model_text = f" for **{model_name}**" if model_name else ""
        st.markdown(f"""
        <div class="notification-card notification-error">
            <div class="notification-content">
                <div class="notification-icon">{icon}</div>
                <div class="notification-message">
                    <div class="notification-title">Error{model_text}</div>
                    <div class="notification-description">{error_details.user_message}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        logger.error(f"Model: {model_name}, Error: {error_details.technical_details}")

    @staticmethod
    def show_success(message: str, model_name: str = None):
        model_text = f" for **{model_name}**" if model_name else ""
        st.markdown(f"""
        <div class="notification-card notification-success">
            <div class="notification-content">
                <div class="notification-icon">✅</div>
                <div class="notification-message">
                    <div class="notification-title">Success{model_text}</div>
                    <div class="notification-description">{message}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_warning(message: str, model_name: str = None):
        model_text = f" for **{model_name}**" if model_name else ""
        st.markdown(f"""
        <div class="notification-card notification-warning">
            <div class="notification-content">
                <div class="notification-icon">⚠️</div>
                <div class="notification-message">
                    <div class="notification-title">Warning{model_text}</div>
                    <div class="notification-description">{message}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_info(message: str):
        st.markdown(f"""
        <div class="notification-card notification-info">
            <div class="notification-content">
                <div class="notification-icon">ℹ️</div>
                <div class="notification-message">
                    <div class="notification-description">{message}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# API key handling & model factories
# ----------------------------------------------------------------------
def clean_api_keys() -> Dict[str, bool]:
    keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"]
    cleaned = {}
    for key in keys:
        val = os.getenv(key)
        if val:
            os.environ[key] = val.strip()
            cleaned[key] = True
        else:
            cleaned[key] = False
    return cleaned

API_KEYS_AVAILABLE = clean_api_keys()

def setup_openai() -> Optional[ChatOpenAI]:
    if not API_KEYS_AVAILABLE.get("OPENAI_API_KEY"):
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.5, timeout=30, max_retries=2)

def setup_gemini() -> Optional[ChatGoogleGenerativeAI]:
    if not API_KEYS_AVAILABLE.get("GOOGLE_API_KEY"):
        return None
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, timeout=30, max_retries=2)

def setup_groq() -> Optional[ChatGroq]:
    if not API_KEYS_AVAILABLE.get("GROQ_API_KEY"):
        return None
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, timeout=30, max_retries=2)

def setup_mistral() -> Optional[ChatMistralAI]:
    if not API_KEYS_AVAILABLE.get("MISTRAL_API_KEY"):
        return None
    return ChatMistralAI(model="mistral-small-latest", temperature=0.5, timeout=30, max_retries=2)

# ----------------------------------------------------------------------
# Safe model query with error handling & retry
# ----------------------------------------------------------------------
async def query_model_safe(
    name: str,
    llm,
    messages: List,
    timeout: int = 45,
    retry_manager: RetryManager = None
) -> Dict[str, Any]:
    if retry_manager is None:
        retry_manager = RetryManager(max_retries=3)

    async def _query():
        start = time.perf_counter()
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
        elapsed = round(time.perf_counter() - start, 2)
        text = response.content if hasattr(response, "content") else str(response)
        return {
            "model": name,
            "time": elapsed,
            "response": text,
            "response_length": len(text),
            "success": True,
            "error": None
        }

    try:
        return await retry_manager.execute_with_retry(_query)
    except Exception as e:
        error_details = ErrorHandler.classify_error(e)
        return {
            "model": name,
            "time": 0.0,
            "response": "",
            "response_length": 0,
            "success": False,
            "error": error_details.user_message,
            "error_details": error_details
        }

# ----------------------------------------------------------------------
# Apply runtime parameters safely (temperature, retries) – only if supported
# ----------------------------------------------------------------------
def configure_model_parameters(model, temperature: float, max_retries: int):
    # Temperature
    if hasattr(model, 'temperature'):
        model.temperature = temperature
    # For ChatOpenAI / ChatGroq / ChatMistralAI, max_retries is a common parameter
    if hasattr(model, 'max_retries'):
        model.max_retries = max_retries
    # Some models (like Gemini) accept `max_retries` via request options; we keep it simple.

async def run_models_async(selected_models, messages, timeout, temperature, max_retries):
    for name, model in selected_models:
        configure_model_parameters(model, temperature, max_retries)

    retry_manager = RetryManager(max_retries=max_retries)
    tasks = [query_model_safe(name, model, messages, timeout, retry_manager) for name, model in selected_models]
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = round(time.perf_counter() - start_time, 2)
    return results, total_time

# ----------------------------------------------------------------------
# Synchronous wrapper to safely run async code in Streamlit
# ----------------------------------------------------------------------
def run_async_safe(coro):
    """Create a new event loop each time to avoid 'Event loop is closed' errors."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# ----------------------------------------------------------------------
# Main Streamlit App
# ----------------------------------------------------------------------
def main():
    st.markdown("<h1 class='main-header'>🤖 Multi-Model AI Arena</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enterprise‑grade AI Model Comparison Platform</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("🔑 API Key Status")
        for key, available in API_KEYS_AVAILABLE.items():
            display_name = key.replace('_API_KEY', '').replace('_', ' ').title()
            if available:
                st.success(f"✅ {display_name}")
            else:
                st.error(f"❌ {display_name}")
        st.markdown("---")

        st.subheader("🤖 Select Models")
        use_openai = st.checkbox("OpenAI GPT-4o Mini", value=API_KEYS_AVAILABLE["OPENAI_API_KEY"], disabled=not API_KEYS_AVAILABLE["OPENAI_API_KEY"])
        use_gemini = st.checkbox("Gemini 1.5 Flash", value=API_KEYS_AVAILABLE["GOOGLE_API_KEY"], disabled=not API_KEYS_AVAILABLE["GOOGLE_API_KEY"])
        use_groq = st.checkbox("Groq Llama 3.3 70B", value=API_KEYS_AVAILABLE["GROQ_API_KEY"], disabled=not API_KEYS_AVAILABLE["GROQ_API_KEY"])
        use_mistral = st.checkbox("Mistral Small", value=API_KEYS_AVAILABLE["MISTRAL_API_KEY"], disabled=not API_KEYS_AVAILABLE["MISTRAL_API_KEY"])

        st.markdown("---")
        with st.expander("🔧 Advanced Settings"):
            timeout = st.slider("Timeout (seconds)", 15, 90, 45)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
            max_retries = st.number_input("Max Retries", 1, 5, 3)

    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_area("📝 Enter your prompt", height=150,
                              placeholder="Ask anything to multiple AI models simultaneously...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

    if submit_button and prompt:
        selected_models = []
        if use_openai:
            m = setup_openai()
            if m:
                selected_models.append(("OpenAI GPT-4o Mini", m))
        if use_gemini:
            m = setup_gemini()
            if m:
                selected_models.append(("Gemini 1.5 Flash", m))
        if use_groq:
            m = setup_groq()
            if m:
                selected_models.append(("Groq Llama 3.3 70B", m))
        if use_mistral:
            m = setup_mistral()
            if m:
                selected_models.append(("Mistral Small", m))

        if not selected_models:
            err = ErrorDetails(ErrorType.UNAVAILABLE, "**No Models Selected**\n\nPlease select at least one model to compare.",
                               "No models selected")
            NotificationComponent.show_error(err)
            return

        messages = [SystemMessage(content=f"You are a helpful, accurate, and concise assistant. Use temperature {temperature}."),
                    HumanMessage(content=prompt)]

        with st.spinner("🔄 Running models with automatic retry logic..."):
            # Safely run async code
            results, total_time = run_async_safe(run_models_async(selected_models, messages, timeout, temperature, max_retries))

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if successful:
            NotificationComponent.show_success(f"Successfully completed {len(successful)} model(s) in {total_time} seconds!")
        for f in failed:
            NotificationComponent.show_error(f["error_details"], f["model"])

        tab1, tab2, tab3 = st.tabs(["📊 Comparison Table", "📝 Detailed Responses", "📈 Statistics"])
        with tab1:
            if successful:
                data = []
                for r in results:
                    preview = (r["response"][:150] + "...") if r["success"] and len(r["response"]) > 150 else (r["response"][:150] if r["success"] else "ERROR")
                    data.append({"Status": "✅" if r["success"] else "❌", "Model": r["model"], "Time": f"{r['time']}s",
                                 "Length": f"{r['response_length']:,} chars", "Preview": preview})
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            else:
                st.warning("No successful responses.")
        with tab2:
            for idx, r in enumerate(results):
                with st.expander(f"📌 {r['model']}", expanded=idx == 0 and r["success"]):
                    if r["success"]:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Response Time", f"{r['time']} seconds")
                        c2.metric("Response Length", f"{r['response_length']:,} characters")
                        c3.metric("Status", "✅ Success")
                        st.markdown("**Full Response:**")
                        st.markdown("---")
                        st.write(r['response'])
                        st.markdown("---")
                    else:
                        st.error(f"Failed: {r.get('error', 'Unknown error')}")
        with tab3:
            if successful:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Models", len(results))
                col2.metric("Successful", len(successful))
                col3.metric("Failed", len(failed))
                success_rate = (len(successful)/len(results)*100) if results else 0
                col4.metric("Success Rate", f"{success_rate:.1f}%")
                if successful:
                    avg_time = sum(r["time"] for r in successful) / len(successful)
                    total_chars = sum(r["response_length"] for r in successful)
                    fastest = min(successful, key=lambda x: x["time"])
                    most_verbose = max(successful, key=lambda x: x["response_length"])
                    st.metric("Avg Response Time", f"{avg_time:.2f} seconds")
                    st.metric("Total Characters", f"{total_chars:,}")
                    st.metric("Parallel Execution", f"{total_time:.2f} seconds")
                    st.info(f"⚡ **Fastest Model:** {fastest['model']} ({fastest['time']}s)")
                    st.info(f"📝 **Most Verbose:** {most_verbose['model']} ({most_verbose['response_length']:,} chars)")
            else:
                st.error("No successful queries.")

        if successful:
            st.markdown("---")
            if st.button("💾 Save Results to File"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"arena_results_{timestamp}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\nMULTI-MODEL AI ARENA RESULTS\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nPrompt: {prompt}\nTotal Time: {total_time}s\n\n")
                    for r in successful:
                        f.write(f"\n{'─' * 80}\nModel: {r['model']}\nTime: {r['time']}s\nLength: {r['response_length']} chars\n\nResponse:\n{'-' * 40}\n{r['response']}\n")
                NotificationComponent.show_success(f"Results saved to {filename}")

    elif submit_button and not prompt:
        NotificationComponent.show_warning("Please enter a prompt before running the comparison.")

    if not any(API_KEYS_AVAILABLE.values()):
        NotificationComponent.show_info("""
        **No API Keys Configured**  
        Create a `.env` file in your repository with your API keys, for example:""")

if __name__ == "__main__":
main()
