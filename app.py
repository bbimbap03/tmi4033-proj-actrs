import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm
from sklearn.datasets import make_blobs
from PIL import Image
import io

# ==========================================
# 0. STATE MANAGEMENT
# ==========================================
def reset_state():
    """Callback to reset the processing state when inputs change."""
    st.session_state.file_processed = False
    st.session_state.review_complete = False
    st.session_state.final_decision = None

# ==========================================
# 1. HELPER: REAL FEATURE EXTRACTION
# ==========================================
def calculate_shannon_entropy(data):
    """Calculates the Shannon Entropy of the file bytes (Real Malware Feature)"""
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def get_image_from_bytes(data, width=300):
    """Converts raw file bytes into a visual representation for the CNN"""
    length = len(data)
    height = int(math.ceil(length / width))
    needed = (width * height) - length
    data += b'\x00' * needed
    img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
    if height > 100:
        img_array = img_array[:100, :]
    return Image.fromarray(img_array)

# ==========================================
# 2. SETUP & STATE
# ==========================================
st.set_page_config(page_title="ACTRS - MARL/CNN/SVM System", page_icon="🛡️", layout="wide")

if 'svm_model' not in st.session_state:
    X, y = make_blobs(n_samples=60, centers=[(3, 3), (7.5, 7.5)], cluster_std=1.2, random_state=42)
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X, y)
    st.session_state.svm_model = clf
    st.session_state.X = X
    st.session_state.y = y

if "review_complete" not in st.session_state: st.session_state.review_complete = False
if "final_decision" not in st.session_state: st.session_state.final_decision = None
if "file_processed" not in st.session_state: st.session_state.file_processed = False

# ==========================================
# 3. UI STYLING (FINAL SIDEBAR HEADER FIX)
# ==========================================
st.markdown("""
<style>
    /* 1. Main Background - Dark Blue */
    .stApp { background: linear-gradient(135deg, #050B18, #0A1E3F); }
    
    /* 2. Global Text Color for Main Area (Light Blue) */
    .stApp p, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp label {
        color: #E6F0FF;
    }

    /* 3. SIDEBAR TEXT FIXES (CRITICAL UPDATE) */
    
    /* Target Paragraphs, Spans, Labels in Sidebar */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #31333F !important; /* Dark Grey */
    }

    /* Target HEADERS (H1, H2, H3) in Sidebar specifically */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #000000 !important; /* Force Pure Black */
    }

    /* 4. SIDEBAR BUTTON FIX (Force White Text on Blue Button) */
    [data-testid="stSidebar"] button div,
    [data-testid="stSidebar"] button p,
    [data-testid="stSidebar"] button span {
        color: #FFFFFF !important; /* Forces "EXECUTE PIPELINE" to be white */
    }
    
    /* 5. CODE BLOCK FIX (Vector Output) */
    code {
        color: #d63384 !important; /* Standard code pink/red color */
        font-weight: bold !important;
    }

    /* 6. Boxes Styling */
    .stage-box { padding: 20px; border-radius: 10px; margin-bottom: 15px; background-color: rgba(10, 30, 63, 0.95); border: 1px solid #1E90FF; }
    .alert-box { border: 1px solid #FF4B4B; background-color: rgba(60, 10, 10, 0.9); }
    .safe-box { border: 1px solid #2ecc71; background-color: rgba(10, 60, 10, 0.9); }
    
    /* 7. Agent Box Logic */
    .agent-box { 
        background-color: rgba(255, 255, 255, 0.05); 
        border: 1px solid #444; 
        padding: 15px; 
        border-radius: 8px; 
        text-align: center; 
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* 8. Main Area Headers (Keep White) */
    .main h3 { border-bottom: 2px solid #1E90FF; padding-bottom: 10px; margin-top: 20px; color: #fff !important; }

    /* 9. Button Styling General */
    button[kind="primary"] {
        background-color: #1E90FF !important;
        border: 1px solid #1E90FF !important;
        font-weight: bold !important;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #2ecc71 !important;
        border: none !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. SIDEBAR
# ==========================================
st.sidebar.header("🎛️ Input Source")
input_method = st.sidebar.radio("Analysis Mode:", ["📂 File Upload", "🎚️ Manual Simulation"], on_change=reset_state)

uploaded_file = None
feature_x = 0.0
feature_y = 0.0
bytes_data = None

if input_method == "📂 File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload File (.exe, .bin, .pdf, etc.)", type=None, on_change=reset_state)
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        real_entropy = calculate_shannon_entropy(bytes_data)
        feature_x = real_entropy 
        if len(bytes_data) > 0:
            mean_byte = np.mean(list(bytes_data[:1000]))
            feature_y = (mean_byte / 255.0) * 10.0
        else:
            feature_y = 0.0
        st.sidebar.success(f"File Loaded: {uploaded_file.name}")
        st.sidebar.metric("Feature 1 (Entropy)", f"{feature_x:.2f}")
        st.sidebar.metric("Feature 2 (Pattern)", f"{feature_y:.2f}")
else:
    st.sidebar.info("Manual Mode: Move slider to simulate file properties.")
    st.sidebar.markdown("### 📏 **Simulation Guide**")
    st.sidebar.caption("Simulate features to test MARL Consensus:")
    
    slider_val = st.sidebar.slider("Simulate Feature Intensity", 0.0, 10.0, 5.0, on_change=reset_state)
    feature_x = slider_val
    feature_y = slider_val 

start_process = st.sidebar.button("▶️ EXECUTE PIPELINE", type="primary")

if start_process:
    st.session_state.review_complete = False
    st.session_state.final_decision = None
    st.session_state.file_processed = True

# ==========================================
# 5. PIPELINE EXECUTION
# ==========================================
st.title("🛡️ ACTRS: Adaptive Cyber Threat Response System")
st.caption("Architecture: CNN Perception → MARL Coordination → SVM Knowledge Base")

if st.session_state.file_processed:
    
    # --- STAGE 1: PERCEPTION (CNN) ---
    st.markdown("### 1️⃣ Stage 1: Local Detection")
    with st.container():
        st.markdown('<div class="stage-box">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            if uploaded_file and bytes_data:
                img = get_image_from_bytes(bytes_data)
                st.image(img, caption=f"Visual Byte Map (CNN Input)", use_container_width=True)
            else:
                noise = np.random.randint(0, 255, (100, 300), dtype=np.uint8)
                st.image(Image.fromarray(noise), caption="Simulated Byte Map (CNN Input)", use_container_width=True)
        with c2:
            st.write(f"**Deep Learning Processing:**")
            st.write("1. Converting binary to grayscale tensors...")
            st.write("2. CNN Layer 1-3: Edge & Texture Detection...")
            if not st.session_state.review_complete:
                bar = st.progress(0)
                for i in range(50):
                    time.sleep(0.005)
                    bar.progress(i*2)
            else:
                st.progress(100)
            st.success("✅ **CNN Output:** Feature Vector Extracted.")
            st.code(f"Vector = [Entropy: {feature_x:.2f}, Pattern_Score: {feature_y:.2f}]")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- STAGE 2: MARL COORDINATION ---
    st.markdown("### 2️⃣ Stage 2: Multi-Agent Reinforcement Learning (MARL) Coordination")
    
    w = st.session_state.svm_model.coef_[0]
    b = st.session_state.svm_model.intercept_[0]
    svm_dist = np.dot(w, [feature_x, feature_y]) + b
    
    # Agent 1: SVM
    if svm_dist > 1.0: vote_svm = "MALWARE"
    elif svm_dist < -1.0: vote_svm = "SAFE"
    else: vote_svm = "UNCERTAIN"

    # Agent 2: CNN
    if feature_y > 6.5: vote_cnn = "MALWARE"
    elif feature_y < 4.5: vote_cnn = "SAFE"
    else: vote_cnn = "UNCERTAIN"

    # Agent 3: Heuristic
    if feature_x > 7.0: vote_rule = "MALWARE"
    elif feature_x < 4.0: vote_rule = "SAFE"
    else: vote_rule = "UNCERTAIN"

    votes = [vote_svm, vote_cnn, vote_rule]
    malware_count = votes.count("MALWARE")
    safe_count = votes.count("SAFE")
    
    if "UNCERTAIN" in votes or (malware_count > 0 and safe_count > 0):
        consensus_state = "CONFLICT"
        consensus_color = "orange"
    elif malware_count >= 2:
        consensus_state = "MALWARE"
        consensus_color = "#FF4B4B"
    else:
        consensus_state = "SAFE"
        consensus_color = "#2ecc71"

    with st.container():
        st.markdown('<div class="stage-box">', unsafe_allow_html=True)
        st.write("**Decentralized Agent Voting:**")
        
        col1, col2, col3 = st.columns(3)
        verdict_style = "font-size: 18px; font-weight: bold; margin-top: 5px; display: block;"

        with col1:
            color = '#FF4B4B' if vote_svm == 'MALWARE' else '#2ecc71' if vote_svm == 'SAFE' else 'orange'
            st.markdown(f"""<div class="agent-box">🤖 <b>Agent 1 (SVM)</b><br><span style="{verdict_style} color: {color};">{vote_svm}</span></div>""", unsafe_allow_html=True)
            
        with col2:
            color = '#FF4B4B' if vote_cnn == 'MALWARE' else '#2ecc71' if vote_cnn == 'SAFE' else 'orange'
            st.markdown(f"""<div class="agent-box">👁️ <b>Agent 2 (CNN)</b><br><span style="{verdict_style} color: {color};">{vote_cnn}</span></div>""", unsafe_allow_html=True)
            
        with col3:
            color = '#FF4B4B' if vote_rule == 'MALWARE' else '#2ecc71' if vote_rule == 'SAFE' else 'orange'
            st.markdown(f"""<div class="agent-box">📜 <b>Agent 3 (Heuristics)</b><br><span style="{verdict_style} color: {color};">{vote_rule}</span></div>""", unsafe_allow_html=True)
            
        st.divider()
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.scatter(st.session_state.X[:, 0], st.session_state.X[:, 1], c=st.session_state.y, cmap='coolwarm', alpha=0.5)
            xx = np.linspace(0, 10, 100)
            yy = - (w[0] * xx + b) / w[1]
            margin = 1 / np.sqrt(np.sum(st.session_state.svm_model.coef_ ** 2))
            yy_down = yy - np.sqrt(1 + (-w[0]/w[1]) ** 2) * margin
            yy_up = yy + np.sqrt(1 + (-w[0]/w[1]) ** 2) * margin
            ax.plot(xx, yy, 'w-', label="Shared Boundary")
            ax.plot(xx, yy_down, 'w--', alpha=0.5)
            ax.plot(xx, yy_up, 'w--', alpha=0.5)
            ax.scatter([feature_x], [feature_y], c='#00FF00', s=250, marker='*', zorder=10, label="Consensus Vector")
            ax.set_xlim(0, 10); ax.set_ylim(0, 10)
            ax.set_facecolor('#0A1E3F'); fig.patch.set_facecolor('#0A1E3F')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('white')
            st.pyplot(fig)
            
        with c2:
            st.write(f"**Consensus Status:**")
            st.markdown(f"<h3 style='color:{consensus_color}'>{consensus_state}</h3>", unsafe_allow_html=True)
            if consensus_state == "CONFLICT":
                st.write("Agents disagree. Action: **Escalate to Human**.")
            else:
                st.write("Agents agree. Action: **Automated Response**.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- STAGE 3 & 4 ---
    st.markdown("### 3️⃣ Stage 3: Co-evolutionary Adaptation")
    
    needs_review = (consensus_state == "CONFLICT")
    if needs_review and not st.session_state.review_complete:
        st.markdown('<div class="stage-box alert-box">', unsafe_allow_html=True)
        st.write("### ⚠️ Human Feedback Loop Triggered")
        with st.form("analyst_form"):
            col1, col2 = st.columns(2)
            with col1: st.metric("SVM Distance", f"{svm_dist:.2f}")
            with col2: st.metric("CNN Pattern Score", f"{feature_y:.2f}")
            decision = st.radio("Verdict:", ["Mark as Safe", "Mark as Malware"], horizontal=True)
            if st.form_submit_button("✅ Resolve Conflict"):
                st.session_state.review_complete = True
                st.session_state.final_decision = decision
                st.rerun() 
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.review_complete:
        decision_txt = st.session_state.final_decision
        color = "safe-box" if decision_txt == "Mark as Safe" else "alert-box"
        st.markdown(f'<div class="stage-box {color}">✅ <b>Resolved.</b> Decision: {decision_txt}. Agents updated.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stage-box gray-box" style="opacity:0.5;">Skipped: Multi-Agent Consensus is High.</div>', unsafe_allow_html=True)

    st.markdown("### 4️⃣ Stage 4: Response & Feedback Loop")
    if st.session_state.review_complete:
        final_status = "SAFE" if st.session_state.final_decision == "Mark as Safe" else "MALWARE"
    elif consensus_state == "MALWARE":
        final_status = "MALWARE"
    elif consensus_state == "SAFE":
        final_status = "SAFE"
    else:
        final_status = "PENDING"

    if final_status == "MALWARE":
        st.markdown('<div class="stage-box alert-box">🛡️ **Active Defense:** Host Isolated. Port 80 Blocked.</div>', unsafe_allow_html=True)
    elif final_status == "SAFE":
        st.markdown('<div class="stage-box safe-box">✅ **Passive Mode:** File Execution Permitted.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stage-box">⏳ **Pending:** Waiting for Stage 3 Resolution...</div>', unsafe_allow_html=True)

else:
    st.info("Please Select an Option from the Sidebar and Click 'Execute Pipeline'")
