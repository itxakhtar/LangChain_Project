import streamlit as st
import os
import asyncio
import time
import logging
import traceback
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from dotenv import load_dotenv  # Fixed: changed from load_dotload_dotenv to load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_arena.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()  # Now this will work correctly

# Page configuration
st.set_page_config(
    page_title="Multi-Model AI Arena",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional notifications
st.markdown("""
<style>
    /* Notification Cards */
    .notification-card {
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(-20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .notification-error {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 5px solid #f56565;
        color: white;
    }
    
    .notification-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-left: 5px solid #ed8936;
        color: white;
    }
    
    .notification-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-left: 5px solid #4299e1;
        color: white;
    }
    
    .notification-success {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        border-left: 5px solid #48bb78;
        color: white;
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .notification-icon {
        font-size: 32px;
    }
    
    .notification-message {
        flex: 1;
    }
    
    .notification-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 5px;
    }
    
    .notification-description {
        font-size: 14px;
        opacity: 0.95;
    }
    
    .retry-button {
        background-color: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 8px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .retry-button:hover {
        background-color: rgba(255,255,255,0.3);
        transform: translateY(-2px);
    }
    
    /* Model Cards */
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .status-success {
        background-color: #c6f6d5;
        color: #22543d;
    }
    
    .status-error {
        background-color: #fed7d7;
        color: #742a2a;
    }
    
    .status-warning {
        background-color: #feebc8;
        color: #7b341e;
    }
    
    /* Loading Spinner */
    .custom-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Error types enumeration
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
    """Structured error details"""
    error_type: ErrorType
    user_message: str
    technical_details: str
    retry_after: Optional[int] = None
    next_available_time: Optional[datetime] = None

class ErrorHandler:
    """Professional error handling with user-friendly messages"""
    
    ERROR_MESSAGES = {
        ErrorType.RATE_LIMIT: {
            "title": "Rate Limit Reached",
            "message": "You've reached the rate limit for this model.",
            "action": "Please wait a moment before trying again."
        },
        ErrorType.AUTHENTICATION: {
            "title": "Authentication Failed",
            "message": "Unable to authenticate with the API.",
            "action": "Please check your API key and try again."
        },
        ErrorType.TIMEOUT: {
            "title": "Request Timeout",
            "message": "The request took too long to complete.",
            "action": "The model is currently busy. Please try again."
        },
        ErrorType.CONNECTION: {
            "title": "Connection Error",
            "message": "Unable to connect to the service.",
            "action": "Please check your internet connection and try again."
        },
        ErrorType.QUOTA_EXCEEDED: {
            "title": "Quota Exceeded",
            "message": "You've exceeded your API quota.",
            "action": "Please check your usage limits or upgrade your plan."
        },
        ErrorType.MAINTENANCE: {
            "title": "Under Maintenance",
            "message": "This model is currently undergoing maintenance.",
            "action": "Please try again later or use another model."
        },
        ErrorType.UNAVAILABLE: {
            "title": "Temporarily Unavailable",
            "message": "The service is currently unavailable.",
            "action": "Please try again in a few minutes."
        },
        ErrorType.UNKNOWN: {
            "title": "Unexpected Error",
            "message": "An unexpected error occurred.",
            "action": "Please try again or contact support if the issue persists."
        }
    }
    
    @staticmethod
    def classify_error(error: Exception) -> ErrorDetails:
        """Classify error and generate user-friendly message"""
        error_str = str(error).lower()
        
        # Rate limit errors
        if any(keyword in error_str for keyword in ['rate limit', 'too many requests', 'rate_limit']):
            return ErrorHandler._create_rate_limit_error(error)
        
        # Authentication errors
        elif any(keyword in error_str for keyword in ['authentication', 'api key', 'unauthorized', 'invalid key']):
            return ErrorDetails(
                error_type=ErrorType.AUTHENTICATION,
                user_message=ErrorHandler._format_user_message(ErrorType.AUTHENTICATION),
                technical_details=str(error)
            )
        
        # Timeout errors
        elif any(keyword in error_str for keyword in ['timeout', 'timed out']):
            return ErrorDetails(
                error_type=ErrorType.TIMEOUT,
                user_message=ErrorHandler._format_user_message(ErrorType.TIMEOUT),
                technical_details=str(error)
            )
        
        # Connection errors
        elif any(keyword in error_str for keyword in ['connection', 'network', 'socket']):
            return ErrorDetails(
                error_type=ErrorType.CONNECTION,
                user_message=ErrorHandler._format_user_message(ErrorType.CONNECTION),
                technical_details=str(error)
            )
        
        # Quota errors
        elif any(keyword in error_str for keyword in ['quota', 'exceeded', 'limit reached']):
            return ErrorDetails(
                error_type=ErrorType.QUOTA_EXCEEDED,
                user_message=ErrorHandler._format_user_message(ErrorType.QUOTA_EXCEEDED),
                technical_details=str(error)
            )
        
        # Default unknown error
        else:
            return ErrorDetails(
                error_type=ErrorType.UNKNOWN,
                user_message=ErrorHandler._format_user_message(ErrorType.UNKNOWN),
                technical_details=str(error)
            )
    
    @staticmethod
    def _create_rate_limit_error(error: Exception) -> ErrorDetails:
        """Create rate limit error with wait time calculation"""
        error_str = str(error).lower()
        retry_after = None
        
        # Try to extract retry time from error message
        if 'retry after' in error_str:
            import re
            match = re.search(r'retry after (\d+)', error_str)
            if match:
                retry_after = int(match.group(1))
        
        next_time = datetime.now() + timedelta(seconds=retry_after) if retry_after else None
        
        user_message = ErrorHandler._format_rate_limit_message(retry_after, next_time)
        
        return ErrorDetails(
            error_type=ErrorType.RATE_LIMIT,
            user_message=user_message,
            technical_details=str(error),
            retry_after=retry_after,
            next_available_time=next_time
        )
    
    @staticmethod
    def _format_user_message(error_type: ErrorType) -> str:
        """Format user-friendly message"""
        messages = ErrorHandler.ERROR_MESSAGES[error_type]
        return f"**{messages['title']}**\n\n{messages['message']}\n\n{messages['action']}"
    
    @staticmethod
    def _format_rate_limit_message(retry_after: Optional[int], next_time: Optional[datetime]) -> str:
        """Format rate limit specific message"""
        if next_time:
            time_str = next_time.strftime("%I:%M %p")
            return f"**Rate Limit Reached**\n\nYou've reached your message limit. Please try again at {time_str}."
        elif retry_after:
            return f"**Rate Limit Reached**\n\nPlease wait {retry_after} seconds before trying again."
        else:
            return "**Rate Limit Reached**\n\nYou've reached your message limit. Please try again in a few minutes."

class RetryManager:
    """Automatic retry with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        progress_callback: Optional[Callable] = None,
        **kwargs
    ):
        """Execute function with automatic retry logic"""
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
                
                # Don't retry on authentication or quota errors
                error_details = ErrorHandler.classify_error(e)
                if error_details.error_type in [ErrorType.AUTHENTICATION, ErrorType.QUOTA_EXCEEDED]:
                    raise
                
                if attempt < self.max_retries:
                    # Calculate exponential backoff
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    
                    if progress_callback:
                        progress_callback(f"Error occurred. Retrying in {delay:.1f} seconds...")
                    
                    await asyncio.sleep(delay)
                else:
                    raise last_error

class NotificationComponent:
    """Professional notification component for Streamlit"""
    
    @staticmethod
    def show_error(error_details: ErrorDetails, model_name: str = None):
        """Display error notification"""
        icon_map = {
            ErrorType.RATE_LIMIT: "⏰",
            ErrorType.AUTHENTICATION: "🔑",
            ErrorType.TIMEOUT: "⏱️",
            ErrorType.CONNECTION: "🌐",
            ErrorType.QUOTA_EXCEEDED: "📊",
            ErrorType.MAINTENANCE: "🔧",
            ErrorType.UNAVAILABLE: "⚠️",
            ErrorType.UNKNOWN: "❓"
        }
        
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
        
        # Log technical error internally
        logger.error(f"Model: {model_name}, Error: {error_details.technical_details}")
    
    @staticmethod
    def show_success(message: str, model_name: str = None):
        """Display success notification"""
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
        """Display warning notification"""
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
        """Display info notification"""
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

# Clean API keys
def clean_api_keys() -> Dict[str, bool]:
    """Clean and validate API keys"""
    keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"]
    cleaned_keys = {}
    
    for key in keys:
        env_value = os.getenv(key)
        if env_value:
            cleaned_value = env_value.strip()
            os.environ[key] = cleaned_value
            cleaned_keys[key] = bool(cleaned_value)
        else:
            cleaned_keys[key] = False
    
    return cleaned_keys

API_KEYS_AVAILABLE = clean_api_keys()

# Model setup functions
def setup_openai() -> Optional[ChatOpenAI]:
    if not API_KEYS_AVAILABLE.get("OPENAI_API_KEY"):
        return None
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.5, timeout=30, max_retries=2)
    except Exception as e:
        logger.error(f"OpenAI setup failed: {e}")
        return None

def setup_gemini() -> Optional[ChatGoogleGenerativeAI]:
    if not API_KEYS_AVAILABLE.get("GOOGLE_API_KEY"):
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, timeout=30, max_retries=2)
    except Exception as e:
        logger.error(f"Gemini setup failed: {e}")
        return None

def setup_groq() -> Optional[ChatGroq]:
    if not API_KEYS_AVAILABLE.get("GROQ_API_KEY"):
        return None
    try:
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, timeout=30, max_retries=2)
    except Exception as e:
        logger.error(f"Groq setup failed: {e}")
        return None

def setup_mistral() -> Optional[ChatMistralAI]:
    if not API_KEYS_AVAILABLE.get("MISTRAL_API_KEY"):
        return None
    try:
        return ChatMistralAI(model="mistral-small-latest", temperature=0.5, timeout=30, max_retries=2)
    except Exception as e:
        logger.error(f"Mistral setup failed: {e}")
        return None

async def query_model_safe(
    name: str,
    llm,
    messages: List,
    timeout: int = 45,
    retry_manager: RetryManager = None
) -> Dict[str, Any]:
    """Query model with comprehensive error handling"""
    
    if retry_manager is None:
        retry_manager = RetryManager(max_retries=3)
    
    async def _query():
        start = time.perf_counter()
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=timeout
        )
        elapsed = round(time.perf_counter() - start, 2)
        
        if hasattr(response, "content"):
            text = response.content
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)
        
        return {
            "model": name,
            "time": elapsed,
            "response": text,
            "response_length": len(text),
            "error": None,
            "success": True
        }
    
    try:
        result = await retry_manager.execute_with_retry(_query)
        return result
        
    except Exception as e:
        error_details = ErrorHandler.classify_error(e)
        elapsed = time.perf_counter() - start if 'start' in locals() else 0
        
        return {
            "model": name,
            "time": round(elapsed, 2),
            "response": "",
            "response_length": 0,
            "error": error_details.user_message,
            "error_details": error_details,
            "success": False
        }

async def run_models_async(selected_models, messages, timeout, temperature, max_retries):
    """Run all models with error handling"""
    
    for name, model in selected_models:
        if hasattr(model, 'temperature'):
            model.temperature = temperature
        if hasattr(model, 'max_retries'):
            model.max_retries = max_retries
    
    retry_manager = RetryManager(max_retries=max_retries)
    
    tasks = [
        query_model_safe(name, model, messages, timeout=timeout, retry_manager=retry_manager)
        for name, model in selected_models
    ]
    
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = round(time.perf_counter() - start_time, 2)
    
    return results, total_time

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 Multi-Model AI Arena</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enterprise-grade AI Model Comparison Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🔑 API Key Status")
        for key, available in API_KEYS_AVAILABLE.items():
            if available:
                st.success(f"✅ {key.replace('_API_KEY', '')}")
            else:
                st.error(f"❌ {key.replace('_API_KEY', '')}")
        
        st.markdown("---")
        
        st.subheader("🤖 Select Models")
        use_openai = st.checkbox("OpenAI GPT-4o Mini", value=API_KEYS_AVAILABLE.get("OPENAI_API_KEY", False), disabled=not API_KEYS_AVAILABLE.get("OPENAI_API_KEY", False))
        use_gemini = st.checkbox("Gemini 1.5 Flash", value=API_KEYS_AVAILABLE.get("GOOGLE_API_KEY", False), disabled=not API_KEYS_AVAILABLE.get("GOOGLE_API_KEY", False))
        use_groq = st.checkbox("Groq Llama 3.3 70B", value=API_KEYS_AVAILABLE.get("GROQ_API_KEY", False), disabled=not API_KEYS_AVAILABLE.get("GROQ_API_KEY", False))
        use_mistral = st.checkbox("Mistral Small", value=API_KEYS_AVAILABLE.get("MISTRAL_API_KEY", False), disabled=not API_KEYS_AVAILABLE.get("MISTRAL_API_KEY", False))
        
        st.markdown("---")
        
        with st.expander("🔧 Advanced Settings"):
            timeout = st.slider("Timeout (seconds)", 15, 90, 45)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
            max_retries = st.number_input("Max Retries", 1, 5, 3)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "📝 Enter your prompt",
            height=150,
            placeholder="Ask anything to multiple AI models simultaneously...",
            help="Your prompt will be sent to all selected AI models"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.button("🚀 Run Comparison", type="primary", use_container_width=True)
    
    if submit_button and prompt:
        # Collect selected models
        selected_models = []
        
        if use_openai and API_KEYS_AVAILABLE.get("OPENAI_API_KEY"):
            model = setup_openai()
            if model:
                selected_models.append(("OpenAI GPT-4o Mini", model))
        
        if use_gemini and API_KEYS_AVAILABLE.get("GOOGLE_API_KEY"):
            model = setup_gemini()
            if model:
                selected_models.append(("Gemini 1.5 Flash", model))
        
        if use_groq and API_KEYS_AVAILABLE.get("GROQ_API_KEY"):
            model = setup_groq()
            if model:
                selected_models.append(("Groq Llama 3.3 70B", model))
        
        if use_mistral and API_KEYS_AVAILABLE.get("MISTRAL_API_KEY"):
            model = setup_mistral()
            if model:
                selected_models.append(("Mistral Small", model))
        
        if not selected_models:
            NotificationComponent.show_error(
                ErrorDetails(
                    error_type=ErrorType.UNAVAILABLE,
                    user_message="**No Models Selected**\n\nPlease select at least one model to compare.",
                    technical_details="No models selected by user"
                )
            )
            return
        
        # Prepare messages
        messages = [
            SystemMessage(content=f"You are a helpful, accurate, and concise assistant. Use temperature {temperature}."),
            HumanMessage(content=prompt)
        ]
        
        # Run queries
        with st.spinner("🔄 Running models with automatic retry logic..."):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            results, total_time = loop.run_until_complete(
                run_models_async(selected_models, messages, timeout, temperature, max_retries)
            )
        
        # Display results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        if successful:
            NotificationComponent.show_success(f"Successfully completed {len(successful)} model(s) in {total_time} seconds!")
        
        # Show errors for failed models
        for result in failed:
            if "error_details" in result:
                NotificationComponent.show_error(result["error_details"], result["model"])
            else:
                NotificationComponent.show_error(
                    ErrorDetails(
                        error_type=ErrorType.UNKNOWN,
                        user_message=result.get("error", "An unknown error occurred"),
                        technical_details=result.get("error", "No details available")
                    ),
                    result["model"]
                )
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Comparison Table", "📝 Detailed Responses", "📈 Statistics"])
        
        with tab1:
            if successful:
                comparison_data = []
                for result in results:
                    status = "✅" if result["success"] else "❌"
                    response_preview = result["response"][:150] + "..." if result["success"] and len(result["response"]) > 150 else result["response"][:150] if result["success"] else f"ERROR"
                    
                    comparison_data.append({
                        "Status": status,
                        "Model": result["model"],
                        "Time": f"{result['time']}s",
                        "Length": f"{result['response_length']:,} chars",
                        "Preview": response_preview
                    })
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No successful model responses to display.")
        
        with tab2:
            for idx, result in enumerate(results):
                with st.expander(f"📌 {result['model']}", expanded=idx == 0 and result["success"]):
                    if result["success"]:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Response Time", f"{result['time']} seconds")
                        with col2:
                            st.metric("Response Length", f"{result['response_length']:,} characters")
                        with col3:
                            st.metric("Status", "✅ Success")
                        
                        st.markdown("**Full Response:**")
                        st.markdown("---")
                        st.write(result['response'])
                        st.markdown("---")
                    else:
                        st.error(f"Failed: {result.get('error', 'Unknown error')}")
        
        with tab3:
            if successful:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Models", len(results))
                with col2:
                    st.metric("Successful", f"{len(successful)} ✅")
                with col3:
                    st.metric("Failed", f"{len(failed)} ❌")
                with col4:
                    success_rate = (len(successful)/len(results)*100) if results else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                if successful:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_time = sum(r["time"] for r in successful) / len(successful)
                        st.metric("Avg Response Time", f"{avg_time:.2f} seconds")
                    with col2:
                        total_chars = sum(r["response_length"] for r in successful)
                        st.metric("Total Characters", f"{total_chars:,}")
                    with col3:
                        st.metric("Parallel Execution", f"{total_time:.2f} seconds")
                    
                    fastest = min(successful, key=lambda x: x["time"])
                    most_verbose = max(successful, key=lambda x: x["response_length"])
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"⚡ **Fastest Model:** {fastest['model']} ({fastest['time']}s)")
                    with col2:
                        st.info(f"📝 **Most Verbose:** {most_verbose['model']} ({most_verbose['response_length']:,} chars)")
            else:
                st.error("No successful queries to display statistics.")
        
        # Save results option
        if successful:
            st.markdown("---")
            if st.button("💾 Save Results to File"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"arena_results_{timestamp}.txt"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("MULTI-MODEL AI ARENA RESULTS\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Total Time: {total_time} seconds\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for result in successful:
                        f.write(f"\n{'─' * 80}\n")
                        f.write(f"Model: {result['model']}\n")
                        f.write(f"Time: {result['time']} seconds\n")
                        f.write(f"Response Length: {result['response_length']} characters\n")
                        f.write(f"\nResponse:\n{'-' * 40}\n{result['response']}\n")
                        f.write(f"{'─' * 80}\n")
                
                NotificationComponent.show_success(f"Results saved to {filename}")
    
    elif submit_button and not prompt:
        NotificationComponent.show_warning("Please enter a prompt before running the comparison.")
    
    # Show API key info
    if not any(API_KEYS_AVAILABLE.values()):
        NotificationComponent.show_info("""
        **No API Keys Configured**
        
        To get started, create a `.env` file with your API keys:
        
The app will automatically detect your keys after restart.
""")

if __name__ == "__main__":
    main()