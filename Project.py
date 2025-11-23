import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import time

# --- 1. PAGE CONFIGURATION & CUSTOM CSS (The "Perfect UI") ---
st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism and cleaner UI
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Custom Card Style */
    .css-card {
        border-radius: 20px;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.7);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Metrics Styling */
    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2980b9;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Button Styling */
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING & PREPROCESSING ---

@st.cache_data
def load_data():
    try:
        # Loading the uploaded dataset
        df = pd.read_csv("cleaned_dataset.csv")
        
        # Fallback if specific columns are named differently in the actual CSV
        # Based on user snippet: Area,Location,Bhk,Bath,Balcony,Parking,Furnishing,Property_Type,Age,Price,Price_Lakhs
        
        # Standardizing column names just in case
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # --- 3. LAT/LON MAPPING FOR MAP INTEGRATION ---
    # Since the dataset doesn't have Lat/Lon, we map top locations manually for the demo
    loc_coords = {
        "Whitefield": [12.9698, 77.7500],
        "Sarjapur Road": [12.9166, 77.6737],
        "Electronic City": [12.8399, 77.6770],
        "Koramangala": [12.9352, 77.6245],
        "Indiranagar": [12.9719, 77.6412],
        "Marathahalli": [12.9591, 77.6974],
        "HSR Layout": [12.9121, 77.6446],
        "Jayanagar": [12.9308, 77.5838],
        "Banashankari": [12.9255, 77.5468],
        "Bellandur": [12.9304, 77.6684],
        "Yelahanka": [13.1005, 77.5963],
        "Bannerghatta Road": [12.8958, 77.6018],
        "Kengeri": [12.9082, 77.4871],
        "Rajajinagar": [12.9982, 77.5530],
        "Hebbal": [13.0354, 77.5988],
        "Malleshwaram": [13.0031, 77.5643],
        "Thanisandra": [13.0547, 77.6339],
        "Hennur Road": [13.0600, 77.6400],
        "JP Nagar": [12.9063, 77.5857],
        "KR Puram": [13.0075, 77.7281]
    }

    # Helper for Mock Data (Amenities, Crime, Traffic)
    def get_location_insights(location):
        # Mock logic for demonstration
        import random
        random.seed(len(location)) # Consistent random per location
        
        traffic_levels = ["Low", "Moderate", "High", "Very High"]
        crime_levels = ["Very Low", "Low", "Moderate"]
        
        return {
            "traffic": traffic_levels[len(location) % 4],
            "crime": crime_levels[len(location) % 3],
            "schools": random.randint(2, 10),
            "hospitals": random.randint(1, 5),
            "parks": random.randint(0, 4),
            "malls": random.randint(0, 3)
        }

    # Preprocessing for ML
    le_loc = LabelEncoder()
    le_furn = LabelEncoder()
    le_prop = LabelEncoder()

    # Creating a copy for training
    model_df = df.copy()
    
    # Handling Categorical Data
    # We fit on the whole dataset for the dropdowns, but in production, you'd save these encoders
    model_df['Location'] = le_loc.fit_transform(model_df['Location'].astype(str))
    model_df['Furnishing'] = le_furn.fit_transform(model_df['Furnishing'].astype(str))
    model_df['Property_Type'] = le_prop.fit_transform(model_df['Property_Type'].astype(str))
    
    # Feature Selection
    X = model_df[['Area', 'Location', 'Bhk', 'Bath', 'Balcony', 'Parking', 'Furnishing', 'Property_Type', 'Age']]
    y = model_df['Price_Lakhs'] # Predicting in Lakhs is more numerically stable

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. SIDEBAR & NAVIGATION ---
    st.sidebar.title("🚀 Navigation")
    menu = st.sidebar.radio("Go to", ["Dashboard & Analysis", "Price Predictor", "Calculators"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Created by Sumit Gaware and Meghavi Sisodiya")

    # --- 5. PAGE: DASHBOARD & ANALYSIS ---
    if menu == "Dashboard & Analysis":
        st.markdown("<div class='css-card'><h1>📊 Bengaluru House Price Prediction</h1></div>", unsafe_allow_html=True)
        
        # Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='css-card'><div class='metric-label'>Total Properties</div><div class='metric-value'>{len(df)}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='css-card'><div class='metric-label'>Avg Price (Lakhs)</div><div class='metric-value'>₹{df['Price_Lakhs'].mean():.2f}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='css-card'><div class='metric-label'>Locations</div><div class='metric-value'>{df['Location'].nunique()}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='css-card'><div class='metric-label'>Avg Area (sqft)</div><div class='metric-value'>{df['Area'].mean():.0f}</div></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📍 Property Distribution Map")
            # Generating Map Data
            map_data = []
            for loc, count in df['Location'].value_counts().items():
                if loc in loc_coords:
                    lat, lon = loc_coords[loc]
                    # Add slight jitter to points so they don't overlap perfectly
                    map_data.append({"lat": lat, "lon": lon, "count": count})
            
            if map_data:
                st.map(pd.DataFrame(map_data), zoom=10)
            else:
                st.warning("Map data unavailable for current locations.")

        with col2:
            st.subheader("📈 Locality Summary")
            # Group by Location
            loc_summary = df.groupby('Location')[['Price_Lakhs', 'Area']].mean().sort_values('Price_Lakhs', ascending=False).head(10)
            st.dataframe(loc_summary.style.format("{:.1f}"))

    # --- 6. PAGE: PRICE PREDICTOR ---
    elif menu == "Price Predictor":
        st.markdown("<div class='css-card'><h1>🏡Price Predictor</h1></div>", unsafe_allow_html=True)
        
        col_input, col_res = st.columns([1, 1])
        
        with col_input:
            with st.container():
                st.markdown("### 📝 Property Details")
                
                area = st.number_input("Area (Sq. Ft)", min_value=300, max_value=10000, value=1200)
                location = st.selectbox("Location", options=le_loc.classes_)
                bhk = st.slider("BHK", 1, 6, 2)
                bath = st.slider("Bathrooms", 1, 6, 2)
                balcony = st.slider("Balcony", 0, 4, 1)
                parking = st.selectbox("Parking", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                furnishing = st.selectbox("Furnishing", options=le_furn.classes_)
                prop_type = st.selectbox("Property Type", options=le_prop.classes_)
                age = st.slider("Property Age (Years)", 0, 50, 5)
                
                st.markdown("### 🧠 Model Selection")
                model_choice = st.selectbox("Choose Algorithm", ["Linear Regression", "Lasso Regression", "Ridge Regression", "Random Forest"])

        with col_res:
            st.markdown("### 🔮 Prediction & Insights")
            
            predict_btn = st.button("Predict Price", use_container_width=True)
            
            if predict_btn:
                # Progress bar for effect
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i+1)
                
                # Model Training (On the fly for this demo project)
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Lasso Regression":
                    model = Lasso(alpha=0.1)
                elif model_choice == "Ridge Regression":
                    model = Ridge(alpha=1.0)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                model.fit(X_train, y_train)
                
                # Prepare Input
                loc_enc = le_loc.transform([location])[0]
                furn_enc = le_furn.transform([furnishing])[0]
                prop_enc = le_prop.transform([prop_type])[0]
                
                input_data = np.array([[area, loc_enc, bhk, bath, balcony, parking, furn_enc, prop_enc, age]])
                
                # Predict
                prediction_lakhs = model.predict(input_data)[0]
                prediction_absolute = prediction_lakhs * 100000
                
                # --- RESULT CARD ---
                st.markdown(f"""
                <div class='css-card' style='text-align:center; background-color: #e8f6f3;'>
                    <h3 style='margin:0; color:#16a085;'>Estimated Value</h3>
                    <h1 style='font-size: 3rem; color: #2c3e50;'>₹ {prediction_lakhs:.2f} Lakhs</h1>
                    <p style='color: #7f8c8d;'>₹ {prediction_absolute:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # --- RENTAL & EMI CARD ---
                rental_yield = prediction_absolute * 0.032 / 12 # Approx 3.2% yield
                st.markdown(f"""
                <div class='css-card'>
                    <div style='display:flex; justify-content:space-between;'>
                        <div>
                            <strong>💰 Est. Monthly Rent</strong><br>
                            <span style='font-size:1.5rem; color:#e67e22;'>₹ {rental_yield:,.0f}</span>
                        </div>
                        <div>
                            <strong>🛡️ Security Deposit (10M)</strong><br>
                            <span style='font-size:1.2rem; color:#7f8c8d;'>₹ {rental_yield*10:,.0f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- LOCALITY INSIGHTS ---
                insights = get_location_insights(location)
                st.markdown("### 🏘️ Neighbourhood Analysis")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"🚦 Traffic Index: **{insights['traffic']}**")
                    st.warning(f"🚓 Crime Index: **{insights['crime']}**")
                with c2:
                    st.write(f"🏫 Schools Nearby: **{insights['schools']}**")
                    st.write(f"🏥 Hospitals Nearby: **{insights['hospitals']}**")
                    st.write(f"🌳 Parks Nearby: **{insights['parks']}**")

    # --- 7. PAGE: CALCULATORS ---
    elif menu == "Calculators":
        st.markdown("<div class='css-card'><h1>🧮 Financial Tools</h1></div>", unsafe_allow_html=True)
        
        col_emi, col_dummy = st.columns([1, 1])
        
        with col_emi:
            st.subheader("EMI Calculator")
            loan_amount = st.number_input("Loan Amount (₹)", value=5000000, step=100000)
            interest_rate = st.number_input("Interest Rate (% p.a.)", value=8.5)
            tenure_years = st.slider("Tenure (Years)", 5, 30, 20)
            
            if st.button("Calculate EMI"):
                p = loan_amount
                r = interest_rate / (12 * 100)
                n = tenure_years * 12
                emi = p * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
                
                total_payment = emi * n
                total_interest = total_payment - p
                
                st.markdown(f"""
                <div class='css-card' style='background-color: #fdf2e9;'>
                    <h2 style='text-align:center; color: #d35400;'>₹ {emi:,.0f} / month</h2>
                    <hr>
                    <p><strong>Total Interest:</strong> ₹ {total_interest:,.0f}</p>
                    <p><strong>Total Amount Payable:</strong> ₹ {total_payment:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("Dataset 'cleaned_dataset.csv' not found. Please upload the file.")