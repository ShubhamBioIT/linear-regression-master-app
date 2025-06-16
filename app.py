import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import StringIO
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# Configure page
st.set_page_config(
    page_title="🎯 Linear Regression Master",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show the title above everything
st.markdown("""
<div class="main-header">
    <h1>🎯 Linear Regression Master</h1>
    <p>Interactive tool to understand and visualize linear regression with real-time parameter adjustment</p>
</div>
""", unsafe_allow_html=True)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 10px;
    }
    
    .parameter-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        animation: slideInUp 0.5s ease-out;
    }
    
    .metric-animation {
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    
    @keyframes slideInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes fadeInLeft {
        from {
            transform: translateX(-30px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .animated-card {
        animation: fadeInLeft 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Sample datasets for educational purposes
SAMPLE_DATASETS = {
    "Perfect Linear": """x,y
1,2
2,4
3,6
4,8
5,10
6,12
7,14
8,16
9,18
10,20""",
    
    "Housing Prices": """size,price
500,150000
750,200000
1000,250000
1250,300000
1500,350000
1750,400000
2000,450000
2250,500000
2500,550000
2750,600000""",
    
    "Study Hours vs Scores": """hours,score
1,45
2,51
3,54
4,58
5,65
6,71
7,78
8,82
9,87
10,95""",
    
    "Temperature vs Ice Cream Sales": """temperature,sales
65,200
70,250
75,300
80,400
85,500
90,650
95,800
100,900
105,1000
110,1200"""
}

def create_header():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Linear Regression Master</h1>
        <p>Interactive tool to understand and visualize linear regression with real-time parameter adjustment</p>
    </div>
    """, unsafe_allow_html=True)

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

def create_interactive_plot(X, y, weight, bias, show_residuals=True):
    """Create interactive Plotly visualization"""
    y_pred = weight * X + bias
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Regression Line', 'Residuals Plot', 'Distribution of Residuals', 'Prediction vs Actual'),
        specs=[[{"colspan": 2}, None],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    fig.add_trace(
        go.Scatter(x=X, y=y, mode='markers', name='Data Points',
                  marker=dict(size=10, color='blue', opacity=0.7)),
        row=1, col=1
    )
    x_line = np.linspace(min(X), max(X), 100)
    y_line = weight * x_line + bias
    fig.add_trace(
        go.Scatter(x=x_line, y=y_line, mode='lines', name=f'y = {weight:.2f}x + {bias:.2f}',
                  line=dict(color='red', width=3)),
        row=1, col=1
    )
    if show_residuals and len(X) <= 50:
        for xi, yi, ypi in zip(X, y, y_pred):
            fig.add_trace(
                go.Scatter(x=[xi, xi], y=[yi, ypi], mode='lines',
                          line=dict(color='green', width=1, dash='dash'),
                          showlegend=False, opacity=0.5),
                row=1, col=1
            )
    residuals = y - y_pred
    fig.add_trace(
        go.Scatter(x=X, y=residuals, mode='markers', name='Residuals',
                  marker=dict(size=8, color='orange')),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_trace(
        go.Scatter(x=y, y=y_pred, mode='markers', name='Pred vs Actual',
                  marker=dict(size=8, color='purple')),
        row=2, col=2
    )
    min_val, max_val = min(min(y), min(y_pred)), max(max(y), max(y_pred))
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name='Perfect Fit',
                  line=dict(color='gray', dash='dash')),
        row=2, col=2
    )
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Linear Regression Analysis Dashboard",
        title_x=0.5
    )
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_xaxes(title_text="X", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_xaxes(title_text="Actual", row=2, col=2)
    fig.update_yaxes(title_text="Predicted", row=2, col=2)
    return fig

def create_loss_landscape(X, y, current_weight, current_bias, loss_type='MSE'):
    """Create 3D loss landscape visualization"""
    weight_range = np.linspace(current_weight - 2, current_weight + 2, 20)
    bias_range = np.linspace(current_bias - 5, current_bias + 5, 20)
    W, B = np.meshgrid(weight_range, bias_range)
    losses = np.zeros_like(W)
    for i in range(len(weight_range)):
        for j in range(len(bias_range)):
            y_pred = weight_range[i] * X + bias_range[j]
            if loss_type == 'MSE':
                losses[j, i] = mean_squared_error(y, y_pred)
            else:
                losses[j, i] = mean_absolute_error(y, y_pred)
    fig = go.Figure(data=[go.Surface(z=losses, x=W, y=B, colorscale='Viridis')])
    current_loss = mean_squared_error(y, current_weight * X + current_bias) if loss_type == 'MSE' else mean_absolute_error(y, current_weight * X + current_bias)
    fig.add_trace(go.Scatter3d(
        x=[current_weight], y=[current_bias], z=[current_loss],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Current Position'
    ))
    fig.update_layout(
        title=f'{loss_type} Loss Landscape',
        scene=dict(
            xaxis_title='Weight',
            yaxis_title='Bias',
            zaxis_title=f'{loss_type} Loss'
        ),
        height=600
    )
    return fig

def find_optimal_parameters(X, y, loss_type):
    """Find optimal parameters using different methods"""
    if loss_type in ["MSE", "RMSE", "R²"]:
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        return float(model.coef_[0]), float(model.intercept_)
    else:  # MAE
        # Use 'epsilon_insensitive' for MAE as per scikit-learn documentation
        model = SGDRegressor(loss='epsilon_insensitive', max_iter=10000, tol=1e-3, random_state=42)
        model.fit(X.reshape(-1, 1), y)
        return float(model.coef_[0]), float(model.intercept_)

def calculate_dynamic_ranges(X, y, current_weight=None, current_bias=None):
    """Calculate dynamic ranges for sliders based on data and optimal parameters"""
    try:
        optimal_weight, optimal_bias = find_optimal_parameters(X, y, "MSE")
    except:
        optimal_weight, optimal_bias = 1.0, 0.0
    x_range = np.max(X) - np.min(X)
    y_range = np.max(y) - np.min(y)
    if x_range > 0:
        data_slope = y_range / x_range
        weight_margin = max(abs(data_slope) * 2, abs(optimal_weight) * 2, 5)
        weight_min = optimal_weight - weight_margin
        weight_max = optimal_weight + weight_margin
    else:
        weight_min, weight_max = -10, 10
    y_margin = max(y_range, abs(optimal_bias) * 2, 10)
    bias_min = optimal_bias - y_margin
    bias_max = optimal_bias + y_margin
    weight_min = max(weight_min, -1000)
    weight_max = min(weight_max, 1000)
    bias_min = max(bias_min, -1000)
    bias_max = min(bias_max, 1000)
    return {
        'weight_min': float(weight_min),
        'weight_max': float(weight_max),
        'bias_min': float(bias_min),
        'bias_max': float(bias_max),
        'optimal_weight': float(optimal_weight),
        'optimal_bias': float(optimal_bias)
    }

# --- Add a global Reset button at the top right of the sidebar ---
with st.sidebar:
    if st.button("🔄 Global Reset", key="global_reset"):
        for key in [
            "weight", "bias", "weight_min", "weight_max", "bias_min", "bias_max",
            "optimal_weight", "optimal_bias", "ranges_calculated", "show_edu"
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Sidebar button to show/hide educational content with toggle functionality ---
with st.sidebar:
    if "show_edu" not in st.session_state:
        st.session_state.show_edu = False
    if st.button("📚 About LR & This Tool", key="show_edu_btn"):
        st.session_state.show_edu = not st.session_state.show_edu

def create_educational_content():
    """Create educational explanations with beginner-friendly, visually attractive content"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); padding: 1.5rem; border-radius: 12px; color: #222; margin-bottom: 1rem;">
        <h2 style="color: #764ba2;">📚 About Linear Regression</h2>
        <p style="font-size: 1.1rem;">
            Linear regression is one of the simplest and most powerful tools in statistics and machine learning. 
            It helps us understand and predict relationships between two variables by fitting a straight line to data.
        </p>
        <ul style="font-size: 1.1rem;">
            <li>Used in <b>finance</b> to predict stock prices</li>
            <li>Applied in <b>healthcare</b> to estimate patient outcomes</li>
            <li>Helps in <b>marketing</b> to forecast sales</li>
            <li>And much more!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔎 What is Linear Regression? (Super Simple!)", expanded=False):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); padding: 1.5rem; border-radius: 12px; color: #222;">
        <h3 style="color: #764ba2;">Linear Regression in a Nutshell</h3>
        <ul style="font-size: 1.1rem;">
            <li><b>Imagine you have a bunch of dots on a graph.</b></li>
            <li>Linear regression draws the <b>straightest line</b> that goes as close as possible to all those dots.</li>
            <li>This line helps you <b>predict</b> new values!</li>
        </ul>
        <hr>
        <h4 style="color: #667eea;">🧮 The Magic Formula:</h4>
        <div style="font-size: 1.2rem; background: #f3e9d2; border-radius: 8px; padding: 0.7rem; margin-bottom: 0.7rem;">
            <b>y = weight × x + bias</b>
        </div>
        <ul>
            <li><b>Weight (Slope):</b> How slanted the line is. Bigger = steeper!</li>
            <li><b>Bias (Intercept):</b> Where the line crosses the y-axis (when x=0).</li>
        </ul>
        <hr>
        <h4 style="color: #764ba2;">🤔 Why Use Linear Regression?</h4>
        <ul>
            <li>To <b>predict</b> things (like house prices, exam scores, sales, etc.)</li>
            <li>To <b>see relationships</b> between two things (like hours studied vs. marks scored)</li>
        </ul>
        <hr>
        <h4 style="color: #764ba2;">📈 More Insights</h4>
        <ul>
            <li>Linear regression is the foundation for more advanced models (like logistic regression, polynomial regression, etc.)</li>
            <li>It assumes a <b>linear relationship</b> between variables (straight line, not curves)</li>
            <li>It's widely used because it's <b>interpretable</b> and <b>fast</b></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🎯 How to Use This Tool (Step-by-Step)", expanded=False):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 12px; color: #222;">
        <ol style="font-size: 1.1rem;">
            <li><b>Pick a dataset</b> (or upload your own CSV file!)</li>
            <li><b>Choose a loss function</b> (explained below!)</li>
            <li>Click <b>"Find Optimal"</b> to let the computer find the best line for you</li>
            <li><b>Move the sliders</b> to see how changing the line affects the fit</li>
            <li>Watch the <b>metrics and plots</b> update instantly</li>
            <li>Check the <b>residuals</b> (the green lines) to see how far off your predictions are</li>
        </ol>
        <div style="margin-top: 1rem; font-size: 1rem;">
            <b>Tip:</b> Try to get the <span style="color: #28a745;">Fit Score</span> as close to <b>100</b> as possible!
        </div>
        <hr>
        <h4 style="color: #764ba2;">🛠️ Extra Tips</h4>
        <ul>
            <li>Try uploading your own data to see how the model fits!</li>
            <li>Experiment with different loss functions to understand their impact</li>
            <li>Use the "Loss Landscape" to visualize how the error changes with different parameters</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🧩 Loss Functions Explained (No Jargon!)", expanded=False):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fcb69f 0%, #ffecd2 100%); padding: 1.5rem; border-radius: 12px; color: #222;">
        <ul style="font-size: 1.1rem;">
            <li><b>🎯 MSE (Mean Squared Error):</b> 
                <ul>
                    <li>Measures how far off your predictions are, but <b>big mistakes count a LOT</b> (because errors are squared)</li>
                    <li>Lower is better!</li>
                </ul>
            </li>
            <li><b>📏 MAE (Mean Absolute Error):</b>
                <ul>
                    <li>Just the average of how wrong you are (no squaring)</li>
                    <li>Every mistake counts the same</li>
                </ul>
            </li>
            <li><b>🧮 RMSE (Root Mean Squared Error):</b>
                <ul>
                    <li>Like MSE, but the answer is in the <b>same units as your data</b></li>
                </ul>
            </li>
            <li><b>🌟 R² (R-squared):</b>
                <ul>
                    <li>Tells you how much of the data is explained by your line</li>
                    <li><b>1.0</b> means perfect fit, <b>0</b> means no fit at all</li>
                </ul>
            </li>
        </ul>
        <div style="margin-top: 1rem; font-size: 1rem;">
            <b>Still confused?</b> Just try each one and see how the line and numbers change!
        </div>
        <hr>
        <h4 style="color: #764ba2;">📚 More on Loss Functions</h4>
        <ul>
            <li>MSE is sensitive to outliers (big errors hurt more)</li>
            <li>MAE is more robust to outliers</li>
            <li>R² is a measure of how well your model explains the data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("💡 What Are Residuals? (The Green Lines)", expanded=False):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #b5ead7 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 12px; color: #222;">
        <ul style="font-size: 1.1rem;">
            <li><b>Residual = Actual Value - Predicted Value</b></li>
            <li>The <b>green lines</b> show how far off each prediction is from the real value</li>
            <li>Shorter green lines = better fit!</li>
        </ul>
        <div style="margin-top: 1rem; font-size: 1rem;">
            <b>Goal:</b> Make the green lines as short as possible!
        </div>
        <hr>
        <h4 style="color: #764ba2;">🔬 Why Residuals Matter</h4>
        <ul>
            <li>They help you spot patterns your model missed</li>
            <li>Randomly scattered residuals = good model</li>
            <li>Patterns in residuals = maybe try a different model!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🚀 Why Learn Linear Regression?", expanded=False):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); padding: 1.5rem; border-radius: 12px; color: #222;">
        <ul style="font-size: 1.1rem;">
            <li>It's the <b>first step</b> in machine learning and data science</li>
            <li>Used everywhere: business, science, sports, health, and more!</li>
            <li>Helps you <b>understand patterns</b> and <b>make predictions</b></li>
            <li>Super useful for <b>school projects, research, and real jobs</b></li>
            <li>Forms the basis for more complex models (like neural networks!)</li>
        </ul>
        <div style="margin-top: 1rem; font-size: 1rem;">
            <b>Master this, and you're on your way to becoming a data wizard! 🧙‍♂️</b>
        </div>
        </div>
        """, unsafe_allow_html=True)

# Show educational content only if sidebar button is toggled on
if st.session_state.show_edu:
    create_educational_content()

def main():
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("#### 📊 Dataset Selection")
        dataset_type = st.selectbox(
            "Choose Dataset",
            ["Upload Custom", "Perfect Linear", "Housing Prices", "Study Hours vs Scores", "Temperature vs Ice Cream Sales"]
        )
        if dataset_type == "Upload Custom":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                st.info("Upload a CSV file or select a sample dataset")
                return
        else:
            df = pd.read_csv(StringIO(SAMPLE_DATASETS[dataset_type]))
        st.markdown("#### 📈 Loss Function")
        loss_type = st.selectbox(
            "Select Loss Function",
            ["MSE", "MAE", "RMSE", "R²"],
            help="Choose the metric to optimize"
        )
        st.markdown("#### 🎨 Visualization Options")
        show_residuals = st.checkbox("Show Residual Lines", value=True)
        show_loss_landscape = st.checkbox("Show Loss Landscape", value=False)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        if len(df.columns) != 2:
            st.error("❌ Please ensure your CSV has exactly 2 columns (X and y)")
            return
        X = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        st.markdown("### 📊 Data Statistics")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: #222;">
                <h4>Dataset Info</h4>
                <p><strong>Points:</strong> {len(X)}</p>
                <p><strong>X Range:</strong> {X.min():.2f} - {X.max():.2f}</p>
                <p><strong>Y Range:</strong> {y.min():.2f} - {y.max():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        with stats_col2:
            correlation = np.corrcoef(X, y)[0, 1]
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ffe082 0%, #ffb300 100%); color: #222;">
                <h4>Correlation</h4>
                <p><strong>r:</strong> {correlation:.3f}</p>
                <p><strong>Strength:</strong> {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}</p>
            </div>
            """, unsafe_allow_html=True)

    with col1:
        if 'weight' not in st.session_state:
            st.session_state.weight = 1.0
        if 'bias' not in st.session_state:
            st.session_state.bias = 0.0
        if 'ranges_calculated' not in st.session_state:
            st.session_state.ranges_calculated = False
        if not st.session_state.ranges_calculated or st.button("🔄 Recalculate Ranges", key="recalc_ranges"):
            ranges = calculate_dynamic_ranges(X, y)
            st.session_state.weight_min = ranges['weight_min']
            st.session_state.weight_max = ranges['weight_max']
            st.session_state.bias_min = ranges['bias_min']
            st.session_state.bias_max = ranges['bias_max']
            st.session_state.optimal_weight = ranges['optimal_weight']
            st.session_state.optimal_bias = ranges['optimal_bias']
            st.session_state.ranges_calculated = True
            st.success(f"🎯 Ranges calculated! Optimal: Weight={ranges['optimal_weight']:.2f}, Bias={ranges['optimal_bias']:.2f}")

        # Only show parameter container (and buttons) if ranges are calculated
        if st.session_state.ranges_calculated:
            button_col1, button_col2, button_col3 = st.columns(3)
            with button_col1:
                if st.button("🎯 Find Optimal", use_container_width=True, key="find_optimal"):
                    with st.spinner("🔍 Calculating optimal parameters..."):
                        time.sleep(0.5)
                        optimal_weight, optimal_bias = find_optimal_parameters(X, y, loss_type)
                        steps = 20
                        current_w, current_b = st.session_state.weight, st.session_state.bias
                        for i in range(steps + 1):
                            alpha = i / steps
                            new_w = current_w + alpha * (optimal_weight - current_w)
                            new_b = current_b + alpha * (optimal_bias - current_b)
                            st.session_state.weight = new_w
                            st.session_state.bias = new_b
                            if i < steps:
                                time.sleep(0.02)
                        st.success(f"✅ Optimal found! Weight: {optimal_weight:.3f}, Bias: {optimal_bias:.3f}")
                        st.rerun()
        with button_col2:
            if st.button("🔄 Reset", use_container_width=True, key="reset_params"):
                st.session_state.weight = 1.0
                st.session_state.bias = 0.0
                st.success("🔄 Parameters reset!")
                st.rerun()
        with button_col3:
            if st.button("🎲 Random", use_container_width=True, key="random_params"):
                weight_range = st.session_state.weight_max - st.session_state.weight_min
                bias_range = st.session_state.bias_max - st.session_state.bias_min
                st.session_state.weight = st.session_state.weight_min + np.random.random() * weight_range
                st.session_state.bias = st.session_state.bias_min + np.random.random() * bias_range
                st.success("🎲 Random parameters set!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.info(f"📊 **Current Ranges:** Weight: [{st.session_state.weight_min:.1f}, {st.session_state.weight_max:.1f}], Bias: [{st.session_state.bias_min:.1f}, {st.session_state.bias_max:.1f}]")

        st.markdown("### 🎛️ Parameter Adjustment")
        weight_step = max(0.001, (st.session_state.weight_max - st.session_state.weight_min) / 1000)
        bias_step = max(0.001, (st.session_state.bias_max - st.session_state.bias_min) / 1000)
        weight = st.slider(
            "Weight (Slope) 📈",
            min_value=st.session_state.weight_min,
            max_value=st.session_state.weight_max,
            value=float(st.session_state.weight),
            step=weight_step,
            help=f"Controls the steepness of the line. Optimal: {st.session_state.optimal_weight:.3f}",
            key="weight_slider"
        )
        bias = st.slider(
            "Bias (Intercept) 📍",
            min_value=st.session_state.bias_min,
            max_value=st.session_state.bias_max,
            value=float(st.session_state.bias),
            step=bias_step,
            help=f"Controls where the line crosses the y-axis. Optimal: {st.session_state.optimal_bias:.3f}",
            key="bias_slider"
        )
        st.session_state.weight = weight
        st.session_state.bias = bias

        st.markdown("### 📈 Interactive Visualization")
        plot_placeholder = st.empty()
        with st.spinner("🎨 Updating visualization..."):
            fig = create_interactive_plot(X, y, weight, bias, show_residuals)
            fig.update_layout(
                transition=dict(duration=300, easing="cubic-in-out"),
                hovermode='closest'
            )
            plot_placeholder.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            })

        # --- Real-time Feedback just below visualization ---
        # Always recalculate optimal parameters for the selected loss_type
        optimal_weight, optimal_bias = find_optimal_parameters(X, y, loss_type)
        st.session_state.optimal_weight = optimal_weight
        st.session_state.optimal_bias = optimal_bias

        st.markdown("### 🎮 Real-time Feedback")
        feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
        with feedback_col1:
            weight_status = "🎯 Perfect!" if abs(weight - optimal_weight) < 0.1 else "📈 Close" if abs(weight - optimal_weight) < 1 else "🔄 Adjust"
            st.markdown(f"""
            <div class="metric-card animated-card" style="background: linear-gradient(135deg, #ffd6e0 0%, #fcb69f 100%); color: #222;">
                <h4>Weight Status</h4>
                <p>{weight_status}</p>
            </div>
            """, unsafe_allow_html=True)
        with feedback_col2:
            bias_status = "🎯 Perfect!" if abs(bias - optimal_bias) < 0.1 else "📍 Close" if abs(bias - optimal_bias) < 1 else "🔄 Adjust"
            st.markdown(f"""
            <div class="metric-card animated-card" style="background: linear-gradient(135deg, #b2f7ef 0%, #4ecdc4 100%); color: #222;">
                <h4>Bias Status</h4>
                <p>{bias_status}</p>
            </div>
            """, unsafe_allow_html=True)
        with feedback_col3:
            weight_diff = abs(weight - optimal_weight)
            bias_diff = abs(bias - optimal_bias)
            distance = np.sqrt(weight_diff**2 + bias_diff**2)
            overall_score = max(0, 100 - distance * 10)
            st.markdown(f"""
            <div class="metric-card animated-card" style="background: linear-gradient(135deg, #f3e9d2 0%, #f9d423 100%); color: #222;">
                <h4>Fit Score</h4>
                <p>{overall_score:.1f}/100</p>
            </div>
            """, unsafe_allow_html=True)
        # --- End Real-time Feedback ---

        y_pred = weight * X + bias
        metrics = calculate_metrics(y, y_pred)
        st.markdown("### 📊 Performance Metrics")
        metric_cols = st.columns(4)
        metric_colors = [
            "background: linear-gradient(135deg, #ffb6b9 0%, #fae3d9 100%); color: #222;",
            "background: linear-gradient(135deg, #b5ead7 0%, #c3cfe2 100%); color: #222;",
            "background: linear-gradient(135deg, #f9d423 0%, #fffcf2 100%); color: #222;",
            "background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); color: #222;"
        ]
        for i, (metric_name, value) in enumerate(metrics.items()):
            with metric_cols[i]:
                if metric_name == 'R²':
                    delta_value = "Good" if value > 0.8 else "Fair" if value > 0.5 else "Poor"
                else:
                    delta_value = "Low" if value < np.std(y)/2 else "High"
                st.markdown(f"""
                <div class="metric-card metric-animation animated-card" style="{metric_colors[i]}">
                    <h4 style="margin: 0;">{metric_name}</h4>
                    <p style="font-size: 1.5rem; margin: 0;"><b>{value:.3f}</b></p>
                    <p style="margin: 0; font-size: 1rem;">{delta_value}</p>
                </div>
                """, unsafe_allow_html=True)
        col_current, col_optimal = st.columns(2)
        with col_current:
            st.markdown(f"""
            <div class="info-box animated-card" style="background: linear-gradient(135deg, #f3e9d2 0%, #f9d423 100%); color: #222;">
                <h4>🎯 Current Parameters</h4>
                <p><strong>Weight:</strong> {weight:.3f}</p>
                <p><strong>Bias:</strong> {bias:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_optimal:
            st.markdown(f"""
            <div class="success-box animated-card" style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); color: #222;">
                <h4>⭐ Optimal Parameters</h4>
                <p><strong>Weight:</strong> {optimal_weight:.3f}</p>
                <p><strong>Bias:</strong> {optimal_bias:.3f}</p>
                <p><strong>Distance:</strong> {distance:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

        if show_loss_landscape:
            st.markdown("### 🗻 Loss Landscape")
            landscape_placeholder = st.empty()
            with st.spinner("🏔️ Generating loss landscape..."):
                landscape_fig = create_loss_landscape(X, y, weight, bias, loss_type)
                landscape_fig.update_layout(
                    transition=dict(duration=500, easing="cubic-in-out")
                )
                landscape_placeholder.plotly_chart(landscape_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎯 Linear Regression Master | Built with ❤️ using Streamlit & Plotly</p>
        <p>Perfect for learning and understanding linear regression concepts!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
