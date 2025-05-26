import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from streamlit_lightweight_charts import renderLightweightCharts
from google import genai
from google.genai import types

# Page configuration
st.set_page_config(
    page_title="TSLA Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color constants
COLOR_BULL = 'rgba(38,166,154,0.9)'  # Green
COLOR_BEAR = 'rgba(239,83,80,0.9)'   # Red

# Initialize session state for chart features
if 'show_signals' not in st.session_state:
    st.session_state.show_signals = False
if 'show_support_resistance' not in st.session_state:
    st.session_state.show_support_resistance = False
if 'show_moving_averages' not in st.session_state:
    st.session_state.show_moving_averages = True

# Initialize session state for chatbot
if 'chatbot_open' not in st.session_state:
    st.session_state.chatbot_open = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'trading_chatbot' not in st.session_state:
    st.session_state.trading_chatbot = None

class TradingChatBot:
    def __init__(self):
        self.data_source = "TSLA_data - Sheet1.csv"
        self.cached_data = None
        self.client = None
        self.model = "gemini-2.5-flash-preview-04-17"
        self.setup_genai_client()
        
    def setup_genai_client(self):
        """Initialize the Gemini AI client"""
        try:
            # Use environment variable for API key security
            api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDVjApDk-IU6xFoK60Y3P_iYRNSfowRqOs")
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize AI client: {e}")
    
    def load_and_cache_data(self):
        """Load and cache the trading data from CSV"""
        if self.cached_data is None:
            try:
                self.cached_data = pd.read_csv(self.data_source)
                # Convert datetime column to proper datetime format
                self.cached_data['datetime'] = pd.to_datetime(self.cached_data['datetime'])
                
                # Data preprocessing and summary
                self.cached_data['year'] = self.cached_data['datetime'].dt.year
                self.cached_data['month'] = self.cached_data['datetime'].dt.month
                self.cached_data['day'] = self.cached_data['datetime'].dt.day
                self.cached_data['hour'] = self.cached_data['datetime'].dt.hour
                
                # Calculate price movements
                self.cached_data['price_change'] = self.cached_data['close'] - self.cached_data['open']
                self.cached_data['price_change_pct'] = (self.cached_data['price_change'] / self.cached_data['open']) * 100
                
            except Exception as e:
                st.error(f"Error loading chatbot data: {e}")
                return None
        
        return self.cached_data
    
    def get_data_summary(self):
        """Generate a comprehensive data summary for the AI context"""
        if self.cached_data is None:
            return "No data loaded"
        
        # Basic statistics
        total_records = len(self.cached_data)
        date_range = f"{self.cached_data['datetime'].min()} to {self.cached_data['datetime'].max()}"
        
        # Price statistics
        price_stats = {
            'avg_price': self.cached_data['close'].mean(),
            'max_price': self.cached_data['high'].max(),
            'min_price': self.cached_data['low'].min(),
            'avg_volume': self.cached_data['volume'].mean(),
            'total_volume': self.cached_data['volume'].sum()
        }
        
        # Bullish/Bearish day analysis
        bullish_days = len(self.cached_data[self.cached_data['direction'] == 'LONG'])
        bearish_days = len(self.cached_data[self.cached_data['direction'] == 'SHORT'])
        neutral_days = len(self.cached_data[self.cached_data['direction'].isna() | (self.cached_data['direction'] == '')])
        
        summary = f"""
        TESLA (TSLA) Trading Data Summary:
        - Total Records: {total_records:,}
        - Date Range: {date_range}
        - Bullish Signals (LONG): {bullish_days:,} records
        - Bearish Signals (SHORT): {bearish_days:,} records  
        - Average Closing Price: ${price_stats['avg_price']:.2f}
        - Highest Price: ${price_stats['max_price']:.2f}
        - Lowest Price: ${price_stats['min_price']:.2f}
        - Average Trading Volume: {price_stats['avg_volume']:,.0f}
        """
        
        return summary
    
    def generate_ai_response(self, user_question):
        """Generate AI response with cached data context"""
        if not self.client:
            return "AI client not initialized. Please check your API key."
        
        data_summary = self.get_data_summary()
        
        # Enhanced prompt with cached data
        enhanced_prompt = f"""
        You are a sophisticated trading bot assistant with access to cached TESLA (TSLA) financial data. 
        
        CACHED DATA CONTEXT:
        {data_summary}
        
        USER QUESTION: {user_question}
        
        Instructions:
        1. Answer the user's question using the cached TSLA data provided above
        2. If the question requires specific calculations, perform them based on the data summary
        3. Provide detailed explanations and reasoning
        4. Suggest related trading insights or follow-up questions
        5. Keep responses concise but informative for a chat interface
        
        Be creative and provide actionable trading insights. Focus on:
        - Trend analysis and patterns
        - Risk assessment 
        - Trading opportunities
        - Market behavior insights
        
        Answer in a professional yet conversational tone as a trading expert.
        """
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=enhanced_prompt)]
                )
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )
            
            response_chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                response_chunks.append(chunk.text)
            
            return "".join(response_chunks)
            
        except Exception as e:
            return f"Error generating AI response: {e}"

@st.cache_data
def load_and_process_data():
    """Load and preprocess the TSLA data with technical indicators"""
    try:
        df = pd.read_csv('TSLA_data - Sheet1.csv', parse_dates=['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Calculate technical indicators
        # Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'TSLA_data - Sheet1.csv' is in the correct directory.")
        return None

def calculate_performance_metrics(df):
    """Calculate trading performance metrics"""
    if 'direction' not in df.columns:
        return {}
    
    # Count signals
    long_signals = len(df[df['direction'] == 'LONG'])
    short_signals = len(df[df['direction'] == 'SHORT'])
    total_signals = long_signals + short_signals
    
    # Calculate basic performance metrics
    df_copy = df.copy()
    df_copy['returns'] = df_copy['close'].pct_change()
    
    # Win rate calculation (simplified)
    signals_with_direction = df_copy[df_copy['direction'].isin(['LONG', 'SHORT'])].copy()
    if len(signals_with_direction) > 1:
        # Calculate next period returns for each signal
        signals_with_direction['next_return'] = signals_with_direction['returns'].shift(-1)
        
        # For LONG signals, positive returns are wins
        long_wins = len(signals_with_direction[
            (signals_with_direction['direction'] == 'LONG') & 
            (signals_with_direction['next_return'] > 0)
        ])
        
        # For SHORT signals, negative returns are wins
        short_wins = len(signals_with_direction[
            (signals_with_direction['direction'] == 'SHORT') & 
            (signals_with_direction['next_return'] < 0)
        ])
        
        total_wins = long_wins + short_wins
        win_rate = (total_wins / total_signals * 100) if total_signals > 0 else 0
    else:
        win_rate = 0
    
    # Calculate volatility
    volatility = df_copy['returns'].std() * np.sqrt(252) * 100 if len(df_copy['returns']) > 1 else 0
    
    return {
        'total_signals': total_signals,
        'long_signals': long_signals,
        'short_signals': short_signals,
        'win_rate': win_rate,
        'volatility': volatility
    }

def prepare_chart_data(df):
    """Prepare data for lightweight charts with signals and support/resistance bands"""
    df_chart = df.copy()
    
    # Convert datetime to Unix timestamp
    df_chart['time'] = df_chart['datetime'].astype('int64') // 10**9
    
    # Candlestick data
    candles = []
    for _, row in df_chart.iterrows():
        candles.append({
            'time': int(row['time']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
    
    # Moving averages
    sma20_data = []
    sma50_data = []
    for _, row in df_chart.dropna(subset=['SMA_20']).iterrows():
        sma20_data.append({
            'time': int(row['time']),
            'value': float(row['SMA_20'])
        })
    
    for _, row in df_chart.dropna(subset=['SMA_50']).iterrows():
        sma50_data.append({
            'time': int(row['time']),
            'value': float(row['SMA_50'])
        })
    
    # Prepare signal markers
    signal_markers = []
    for _, row in df_chart.iterrows():
        if pd.notna(row.get('direction', None)) and row['direction'] in ['LONG', 'SHORT']:
            if row['direction'] == 'LONG':
                # Green up arrow below the candle
                signal_markers.append({
                    'time': int(row['time']),
                    'position': 'belowBar',
                    'color': '#00ff00',
                    'shape': 'arrowUp',
                    'text': 'LONG'
                })
            elif row['direction'] == 'SHORT':
                # Red down arrow above the candle
                signal_markers.append({
                    'time': int(row['time']),
                    'position': 'aboveBar',
                    'color': '#ff0000',
                    'shape': 'arrowDown',
                    'text': 'SHORT'
                })
        elif pd.notna(row.get('direction', None)) and row['direction'] not in ['LONG', 'SHORT']:
            # Yellow circle for other directions
            signal_markers.append({
                'time': int(row['time']),
                'position': 'inBar',
                'color': '#ffff00',
                'shape': 'circle',
                'text': str(row['direction'])
            })
      # Prepare support bands - create bands between min and max values
    support_bands_lower = []
    support_bands_upper = []
    if 'Support' in df_chart.columns:
        for _, row in df_chart.iterrows():
            try:
                support_str = row['Support']
                if pd.notna(support_str) and support_str != '':
                    support_levels = eval(support_str) if isinstance(support_str, str) else support_str
                    if isinstance(support_levels, (list, tuple)) and len(support_levels) > 0:
                        support_min = min(support_levels)
                        support_max = max(support_levels)
                        support_bands_lower.append({
                            'time': int(row['time']),
                            'value': support_min
                        })
                        support_bands_upper.append({
                            'time': int(row['time']),
                            'value': support_max
                        })
            except:
                continue
    
    # Prepare resistance bands - create bands between min and max values
    resistance_bands_lower = []
    resistance_bands_upper = []
    if 'Resistance' in df_chart.columns:
        for _, row in df_chart.iterrows():
            try:
                resistance_str = row['Resistance']
                if pd.notna(resistance_str) and resistance_str != '':
                    resistance_levels = eval(resistance_str) if isinstance(resistance_str, str) else resistance_str
                    if isinstance(resistance_levels, (list, tuple)) and len(resistance_levels) > 0:
                        resistance_min = min(resistance_levels)
                        resistance_max = max(resistance_levels)
                        resistance_bands_lower.append({
                            'time': int(row['time']),
                            'value': resistance_min
                        })
                        resistance_bands_upper.append({
                            'time': int(row['time']),
                            'value': resistance_max
                        })
            except:
                continue    
    return candles, sma20_data, sma50_data, signal_markers, support_bands_lower, support_bands_upper, resistance_bands_lower, resistance_bands_upper

def render_chatbot_panel():
    """Render the chatbot as a dedicated panel at the top of the page"""
    
    # Chatbot control buttons
    col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
    
    with col2:
        if st.button("💬 Chat", help="Toggle AI Assistant"):
            st.session_state.chatbot_open = not st.session_state.chatbot_open
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear", help="Clear chat history"):
            st.session_state.chat_messages = []
            st.rerun()
            
    with col4:
        st.write(f"{'🟢 Open' if st.session_state.chatbot_open else '🔴 Closed'}")
    
    # Only show chatbot panel if it's open
    if st.session_state.chatbot_open:
        # Chatbot panel with modern styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
            color: white;
        ">
            <h3 style="margin: 0 0 15px 0; color: white;">🤖 AI Trading Assistant</h3>
            <p style="margin: 0; opacity: 0.9;">Ask me anything about TSLA trading data, patterns, and insights!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick suggestion buttons
        st.markdown("**💡 Quick Questions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        suggestion_questions = [
            ("💹 Price Analysis", "Analyze current price trend and provide insights"),
            ("📊 Volume Analysis", "What does the trading volume tell us about current market sentiment?"),
            ("🎯 Support/Resistance", "Show me the key support and resistance levels"),
            ("📈 Trading Signals", "What trading signals should I watch for?")
        ]
        
        for i, (label, question) in enumerate(suggestion_questions):
            with [col1, col2, col3, col4][i]:
                if st.button(label, key=f"suggest_{i}"):
                    st.session_state.chat_messages.append({"role": "user", "content": question})
                    try:
                        with st.spinner("🤔 Analyzing..."):
                            response = st.session_state.trading_chatbot.generate_ai_response(question)
                            st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
        
        # Chat messages container with scroll
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_messages:
                # Create scrollable chat area
                st.markdown("""
                <div style="
                    max-height: 400px;
                    overflow-y: auto;
                    background: #f8f9fc;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 15px 0;
                    border: 1px solid #e1e5e9;
                ">
                """, unsafe_allow_html=True)
                
                # Display chat messages
                for i, message in enumerate(st.session_state.chat_messages):
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div style="
                            text-align: right;
                            margin: 10px 0;
                        ">
                            <div style="
                                display: inline-block;
                                background: #667eea;
                                color: white;
                                padding: 10px 15px;
                                border-radius: 18px;
                                border-bottom-right-radius: 4px;
                                max-width: 80%;
                                font-size: 14px;
                                line-height: 1.4;
                            ">
                                {message['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            text-align: left;
                            margin: 10px 0;
                        ">
                            <div style="
                                display: inline-block;
                                background: white;
                                color: #374151;
                                padding: 10px 15px;
                                border-radius: 18px;
                                border-bottom-left-radius: 4px;
                                max-width: 80%;
                                font-size: 14px;
                                line-height: 1.4;
                                border: 1px solid #e1e5e9;
                            ">
                                {message['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("👋 Start a conversation by typing a question or clicking a suggestion button above!")
        
        # Chat input form
        with st.form("chatbot_form", clear_on_submit=True):
            st.markdown("**Type your question:**")
            user_input = st.text_area("", placeholder="Ask about TSLA trading patterns, signals, support levels, etc...", height=80, key="chat_input")
            
            col1, col2 = st.columns([8, 2])
            with col2:
                send_button = st.form_submit_button("Send 📤", use_container_width=True)
            
            # Process user input
            if send_button and user_input.strip():
                # Add user message
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                # Generate AI response
                try:
                    with st.spinner("🤔 Analyzing TSLA data..."):
                        response = st.session_state.trading_chatbot.generate_ai_response(user_input)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                
                st.rerun()

def main():
    st.title("🚀 Advanced TSLA Trading Bot")
    
    # Custom CSS for better button styling
    st.markdown("""
    <style>
    .stButton > button {
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if st.session_state.trading_chatbot is None:
        st.session_state.trading_chatbot = TradingChatBot()
        st.session_state.trading_chatbot.load_and_cache_data()
    
    # Render chatbot panel at the top
    render_chatbot_panel()
    
    st.markdown("---")
    
    # Load data
    df = load_and_process_data()
    if df is None:
        return# Sidebar controls
    st.sidebar.header("📊 Trading Controls")
    
    # Chart feature toggles in sidebar
    st.sidebar.subheader("🎛️ Chart Features")
    st.session_state.show_signals = st.sidebar.checkbox("Show Signal Arrows", value=st.session_state.show_signals)
    st.session_state.show_support_resistance = st.sidebar.checkbox("Show Support/Resistance Bands", value=st.session_state.show_support_resistance)
    st.session_state.show_moving_averages = st.sidebar.checkbox("Show Moving Averages", value=st.session_state.show_moving_averages)
    
    # Time range selection
    time_range = st.sidebar.selectbox(
        "Select Time Range",
        ["Last 100 records", "Last 500 records", "Last 1000 records", "All data"]
    )
    
    # Filter data based on selection
    if time_range == "Last 100 records":
        filtered_df = df.tail(100).copy()
    elif time_range == "Last 500 records":
        filtered_df = df.tail(500).copy()
    elif time_range == "Last 1000 records":
        filtered_df = df.tail(1000).copy()
    else:
        filtered_df = df.copy()
      # Display key metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_price = filtered_df['close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        price_change = filtered_df['close'].iloc[-1] - filtered_df['close'].iloc[0]
        price_change_pct = (price_change / filtered_df['close'].iloc[0]) * 100
        st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    
    with col3:
        avg_volume = filtered_df['volume'].mean()
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    # Performance metrics
    st.subheader("📈 Trading Performance")
    metrics = calculate_performance_metrics(filtered_df)
    
    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Signals", metrics['total_signals'])
        with col2:
            st.metric("Long Signals", metrics['long_signals'])
        with col3:
            st.metric("Short Signals", metrics['short_signals'])
        with col4:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        with col5:
            st.metric("Volatility", f"{metrics['volatility']:.2f}%")      # Chart options - Dark theme configuration
    chartOptions = {
        "width": 1000,
        "height": 500,
        "rightPriceScale": {
            "scaleMargins": {
                "top": 0.1,
                "bottom": 0.1,
            },
            "borderVisible": False,
        },
        "layout": {
            "background": {
                "type": 'solid',
                "color": '#131722'
            },
            "textColor": '#d1d4dc',
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(42, 46, 57, 0)',
            },
            "horzLines": {
                "color": 'rgba(42, 46, 57, 0.6)',
            }
        },        "crosshair": {"mode": 0},
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "barSpacing": 16,  # Increased from 10 to 16 (60% more zoom)
            "minBarSpacing": 13,  # Increased from 8 to 13 (60% more zoom)
            "timeVisible": True,
            "secondsVisible": False,
        },
        "watermark": {
            "visible": True,
            "fontSize": 48,
            "horzAlign": 'center',
            "vertAlign": 'center',
            "color": 'rgba(171, 71, 188, 0.3)',
            "text": 'TSLA Bot',
        }
    }    # Prepare chart data - candlestick, moving averages, signals, and support/resistance bands
    candles, sma20_data, sma50_data, signal_markers, support_bands_lower, support_bands_upper, resistance_bands_lower, resistance_bands_upper = prepare_chart_data(filtered_df)

    # Chart control buttons at the top
    st.markdown("### 📊 TSLA Price Chart")
      # Chart control buttons at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎯 Toggle Signals", help="Show/hide trading signal arrows"):
            st.session_state.show_signals = not st.session_state.show_signals
            st.rerun()
    with col2:
        if st.button("📏 Toggle S/R Bands", help="Show/hide support/resistance bands"):
            st.session_state.show_support_resistance = not st.session_state.show_support_resistance
            st.rerun()
    with col3:
        if st.button("📈 Toggle MA Lines", help="Show/hide moving average lines"):
            st.session_state.show_moving_averages = not st.session_state.show_moving_averages
            st.rerun()
    with col4:
        if st.button("🔄 Reset View", help="Reset all features to default"):
            st.session_state.show_signals = True
            st.session_state.show_support_resistance = True
            st.session_state.show_moving_averages = True
            st.rerun()    # Series configurations - candlestick with conditional features
    seriesCandlestickChart = [
        {
            "type": 'Candlestick',
            "data": candles,
            "options": {
                "upColor": COLOR_BULL,
                "downColor": COLOR_BEAR,
                "borderVisible": False,
                "wickUpColor": COLOR_BULL,
                "wickDownColor": COLOR_BEAR
            },
            "markers": signal_markers if st.session_state.show_signals else []
        }
    ]
    
    # Add moving averages conditionally
    if st.session_state.show_moving_averages:
        if sma20_data:
            seriesCandlestickChart.append({
                "type": 'Line',
                "data": sma20_data,
                "options": {
                    "color": 'rgba(255, 165, 0, 0.8)',
                    "lineWidth": 2,
                    "title": "SMA 20"
                }
            })
        
        if sma50_data:
            seriesCandlestickChart.append({
                "type": 'Line',
                "data": sma50_data,
                "options": {
                    "color": 'rgba(75, 0, 130, 0.8)',
                    "lineWidth": 2,
                    "title": "SMA 50"
                }
            })
    
    # Add support bands conditionally (green) - using line series for upper and lower bounds
    if st.session_state.show_support_resistance:
        if support_bands_lower and support_bands_upper:
            seriesCandlestickChart.append({
                "type": 'Line',
                "data": support_bands_lower,
                "options": {
                    "color": 'rgba(0, 255, 0, 0.6)',
                    "lineWidth": 1,
                    "lineStyle": 2,  # Dashed line
                    "title": "Support Lower"
                }
            })
            seriesCandlestickChart.append({
                "type": 'Line',
                "data": support_bands_upper,
                "options": {
                    "color": 'rgba(0, 255, 0, 0.6)',
                    "lineWidth": 1,
                    "lineStyle": 2,  # Dashed line
                    "title": "Support Upper"
                }
            })
        
        # Add resistance bands (red) - using line series for upper and lower bounds
        if resistance_bands_lower and resistance_bands_upper:
            seriesCandlestickChart.append({
                "type": 'Line',
                "data": resistance_bands_lower,
                "options": {
                    "color": 'rgba(255, 0, 0, 0.6)',
                    "lineWidth": 1,
                    "lineStyle": 2,  # Dashed line
                    "title": "Resistance Lower"
                }
            })
            seriesCandlestickChart.append({
                "type": 'Line',
                "data": resistance_bands_upper,
                "options": {
                    "color": 'rgba(255, 0, 0, 0.6)',
                    "lineWidth": 1,
                    "lineStyle": 2,  # Dashed line
                    "title": "Resistance Upper"
                }
            })    # Display current chart features status
    feature_status = []
    if st.session_state.show_signals:
        feature_status.append("🎯 Signal Arrows")
    if st.session_state.show_support_resistance:
        feature_status.append("📏 S/R Bands")
    if st.session_state.show_moving_averages:
        feature_status.append("📈 Moving Averages")
    
    if feature_status:
        st.info(f"**Active Features:** {' | '.join(feature_status)}")
    else:
        st.warning("**All features disabled** - Showing candlesticks only")

    # Render the dark theme candlestick chart with conditional features
    charts_to_render = [
        {"chart": chartOptions, "series": seriesCandlestickChart}
    ]
    renderLightweightCharts(charts_to_render, 'priceAndVolume')
    
    # Trading signals analysis
    st.subheader("🎯 Trading Signals Analysis")
    
    # Signal distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Signal Distribution**")
        signal_counts = filtered_df['direction'].value_counts()
        signal_data = []
        for direction, count in signal_counts.items():
            if direction in ['LONG', 'SHORT']:
                signal_data.append({"Signal": direction, "Count": count})
        
        if signal_data:
            signal_df = pd.DataFrame(signal_data)
            st.dataframe(signal_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Recent Signals**")
        recent_signals = filtered_df[filtered_df['direction'].isin(['LONG', 'SHORT'])].tail(10)
        if len(recent_signals) > 0:
            display_signals = recent_signals[['datetime', 'direction', 'close']].copy()
            display_signals['datetime'] = display_signals['datetime'].dt.strftime('%m-%d %H:%M')
            display_signals['close'] = display_signals['close'].round(2)
            st.dataframe(display_signals, use_container_width=True, hide_index=True)
        else:
            st.info("No recent signals found")
      # Technical analysis summary
    st.subheader("🔍 Technical Analysis Summary")
    
    latest_data = filtered_df.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Moving Averages**")
        if not pd.isna(latest_data['SMA_20']):
            sma20_trend = "🟢 Bullish" if latest_data['close'] > latest_data['SMA_20'] else "🔴 Bearish"
            st.write(f"SMA 20: {sma20_trend}")
        
        if not pd.isna(latest_data['SMA_50']):
            sma50_trend = "🟢 Bullish" if latest_data['close'] > latest_data['SMA_50'] else "🔴 Bearish"
            st.write(f"SMA 50: {sma50_trend}")
    
    with col2:
        st.write("**Price Analysis**")
        if not pd.isna(latest_data['close']):
            current_price = latest_data['close']
            st.write(f"Current Price: ${current_price:.2f}")
            
            # Price trend based on moving averages
            if not pd.isna(latest_data['SMA_20']) and not pd.isna(latest_data['SMA_50']):
                if latest_data['SMA_20'] > latest_data['SMA_50']:
                    trend = "🟢 Uptrend"
                else:
                    trend = "🔴 Downtrend"
                st.write(f"Trend: {trend}")
    
    # Support and Resistance levels
    st.subheader("📏 Support & Resistance Levels")
    
    if 'Support' in filtered_df.columns and 'Resistance' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Support Levels**")
            try:
                support_str = latest_data['Support']
                if pd.notna(support_str) and support_str != '':
                    support_levels = eval(support_str)
                    for i, level in enumerate(support_levels[:5]):
                        st.write(f"S{i+1}: ${level:.2f}")
            except:
                st.write("No support data available")
        
        with col2:
            st.write("**Resistance Levels**")
            try:
                resistance_str = latest_data['Resistance']
                if pd.notna(resistance_str) and resistance_str != '':
                    resistance_levels = eval(resistance_str)
                    for i, level in enumerate(resistance_levels[:5]):
                        st.write(f"R{i+1}: ${level:.2f}")
            except:
                st.write("No resistance data available")
    
    # Display raw data option
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("📋 Raw Data")
        st.dataframe(filtered_df.tail(20), use_container_width=True)

if __name__ == "__main__":
    main()
