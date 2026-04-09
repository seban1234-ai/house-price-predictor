import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Smart House Predictor", layout="wide")

# ================== CSS ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #F8FBFF, #E6F0FA);
    color: #111;
}

h1, h2, h3 {
    color: #0A2A43;
}

p, label, div {
    color: #222 !important;
}

section[data-testid="stSidebar"] {
    background-color: #DCEEFF;
}

.stButton>button {
    background-color: #0077B6;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}

.header {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #0077B6;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================== LOGIN ==================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful")
        else:
            st.error("Invalid Credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ================== SIDEBAR ==================
st.sidebar.title("🏠 Smart Predictor")

menu = st.sidebar.radio("Navigation", ["Prediction", "Analytics", "Upload Data", "About"])

st.sidebar.markdown("---")
st.sidebar.subheader("✨ Features")
st.sidebar.write("""
✔ Multiple ML Models  
✔ Auto Best Model Selection  
✔ Interactive UI  
✔ Data Visualization  
✔ Upload Dataset  
✔ Login System  
""")

# ================== LOAD DATA ==================
try:
    df = pd.read_csv("housing.csv")
except:
    st.error("⚠️ housing.csv not found!")
    st.stop()

# ================== HEADER ==================
st.markdown('<div class="header">🏡 Smart House Predictor</div>', unsafe_allow_html=True)
st.write("### Predict house prices using Machine Learning")

# ================== MODEL ==================
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

results = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results[name] = r2_score(y_test, pred)

best_model_name = max(results, key=results.get)
model = models[best_model_name]

# ================== PREDICTION ==================
if menu == "Prediction":

    st.success(f"✅ Best Model: {best_model_name}")

    col1, col2 = st.columns(2)

    with col1:
        area = st.slider("📏 Area", 500, 5000, 1500)
        bedrooms = st.slider("🛏 Bedrooms", 1, 5, 2)

    with col2:
        bathrooms = st.slider("🛁 Bathrooms", 1, 4, 2)
        age = st.slider("🏚 Age", 0, 50, 10)

    location = st.selectbox("📍 Location", ["Urban", "Suburban", "Rural"])
    location_map = {"Urban": 2, "Suburban": 1, "Rural": 0}
    loc_val = location_map[location]

    if st.button("🔮 Predict Price"):
        prediction = model.predict([[area, bedrooms, bathrooms, age]])

        st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f}")

        if prediction[0] > 500000:
            st.info("🏆 Premium Property")
        elif prediction[0] > 300000:
            st.info("👍 Mid-range Property")
        else:
            st.info("💸 Budget Property")

# ================== ANALYTICS ==================
elif menu == "Analytics":

    st.subheader("🤖 Model Comparison")

    for k, v in results.items():
        st.write(f"{k}: {v:.2f}")

    st.subheader("📊 Price Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Price'], bins=20)
    st.pyplot(fig)

    st.subheader("📈 Area vs Price")
    fig, ax = plt.subplots()
    ax.scatter(df['Area'], df['Price'])
    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ================== UPLOAD ==================
elif menu == "Upload Data":

    file = st.file_uploader("Upload CSV")

    if file:
        new_df = pd.read_csv(file)
        st.write(new_df.head())

# ================== ABOUT ==================
elif menu == "About":

    st.write("""
    ### 👨‍💻 About Project
    
    This is an advanced ML web app that:
    
    ✔ Compares multiple ML models  
    ✔ Uses Gradient Boosting  
    ✔ Shows analytics & insights  
    ✔ Allows dataset upload  
    
    Built using Streamlit + Scikit-learn
    """)

# ================== FOOTER ==================
st.markdown("---")
st.write("🚀 Developed by Sebastian Roy")