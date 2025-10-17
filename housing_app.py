import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# ---------------------------
# Intro Section
# ---------------------------
st.title("🏠 Housing Price Prediction App")
st.markdown("""
Welcome to the **Housing Price Prediction App**!  

This app predicts the **price of a house** based on key features like:  
- **Number of rooms**  
- **Age of the house**  
- **Property tax rate**  
- **Crime rate in the area**  
- **% of lower status population in the area**  

Adjust the sliders in the **sidebar** to input your house’s features, and the app will predict an **estimated house price** (in $1000s) instantly.  

💡 **Tip:** More rooms and a safer neighborhood usually increase the house price!
""")

# ---------------------------
# Show Sample Dataset with Feature Guide
# ---------------------------
st.subheader("📊 Sample Housing Data")

num_rows = st.slider("Select number of rows to view", 5, len(df), 10)
st.dataframe(df.head(num_rows))

st.markdown("""
### 🧾 Feature Guide  

Here are all the features available in the original **Boston Housing dataset**.  
We’re using only the **main beginner-friendly ones** in this app — the rest are **optional (advanced)** features you can explore later.  

#### 🏠 Features Used in This App:
- **rm:** Average number of rooms per house *(more rooms → higher price)*  
- **age:** Age of the house in years *(newer houses → higher price)*  
- **tax:** Property tax rate per $10,000 *(higher tax → lower price)*  
- **crim:** Crime rate by town *(lower crime → higher price)*  
- **lstat:** % of lower status population in the area *(lower → higher price)*  
- **chas:** Charles River dummy variable *(1 = near the river, 0 = not near)*  
- **medv:** Median house value in $1000s *(target we want to predict)*  

#### ⚙️ Optional (Advanced) Features:
- **zn:** Proportion of residential land zoned for large lots  
- **indus:** Proportion of non-retail business acres per town  
- **nox:** Nitric oxide concentration (air pollution level)  
- **dis:** Distance to employment centers (proximity to city)  
- **rad:** Accessibility to radial highways  
- **ptratio:** Pupil–teacher ratio by town  
- **b:** 1000(Bk − 0.63)² — a demographic index (legacy feature)  
""")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("🏡 Predict Your House Price")
st.sidebar.markdown("Adjust the sliders below to input your house's features and see the predicted price instantly.")

def user_input_features():
    rm = st.sidebar.slider('Average number of rooms (rm)', float(df.rm.min()), float(df.rm.max()), float(df.rm.mean()))
    age = st.sidebar.slider('Age of the house (age)', float(df.age.min()), float(df.age.max()), float(df.age.mean()))
    crim = st.sidebar.slider('Crime rate (crim)', float(df.crim.min()), float(df.crim.max()), float(df.crim.mean()))
    lstat = st.sidebar.slider('% lower status population (lstat)', float(df.lstat.min()), float(df.lstat.max()), float(df.lstat.mean()))
    tax = st.sidebar.slider('Property tax rate (tax)', float(df.tax.min()), float(df.tax.max()), float(df.tax.mean()))
    
    # Optional checkbox for extra feature
    chas = st.sidebar.checkbox('Near Charles River? (chas)', value=False)
    
    data = {'rm': rm,
            'age': age,
            'crim': crim,
            'lstat': lstat,
            'tax': tax,
            'chas': int(chas)}  # convert boolean to 0/1
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ---------------------------
# Train Model
# ---------------------------
X = df[['rm','age','crim','lstat','tax','chas']]
y = df['medv']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# ---------------------------
# Model Performance
# ---------------------------
from sklearn.metrics import r2_score, mean_absolute_error

# Predict on all data (for evaluation)
y_pred = model.predict(X_scaled)

# Compute metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

st.subheader("📈 Model Performance")

st.markdown(f"""
- **R² Score:** {r2:.2f}  
  _(Explains how much of the variation in house prices is captured by the model.  
  Closer to 1 means better fit.)_

- **Mean Absolute Error (MAE):** ${mae * 1000:,.0f}  
  _(On average, the prediction differs by this much from actual prices.)_
""")

# Optional: Actual vs Predicted Scatter Plot
fig = px.scatter(x=y, y=y_pred, labels={'x':'Actual Price ($1000s)', 'y':'Predicted Price ($1000s)'},
                 title="Actual vs Predicted Prices")
fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
              line=dict(color='red', dash='dash'))
st.plotly_chart(fig)

# ---------------------------
# Make Prediction
# ---------------------------
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

# Convert to full dollars
predicted_price = prediction[0] * 1000  # $1000s → dollars

# Determine price category and emoji
if predicted_price < 15000:
    emoji = "💸"  # Cheap
    category = "Cheap"
elif predicted_price < 35000:
    emoji = "💵💵"  # Moderate
    category = "Moderate"
else:
    emoji = "🏡✨"  # Expensive
    category = "Expensive"

# Display price with default styling
st.subheader("🏡 Predicted House Price")
st.markdown(f"#### ${predicted_price:,.0f} {emoji} ({category})")

# ---------------------------
# Explain Prediction in Beginner-Friendly Way
# ---------------------------
st.subheader("💡 How does each feature affect the house price?")

st.markdown("""
- **More rooms** → usually increases the house price  
- **Older house (higher age)** → usually decreases the house price  
- **Higher crime rate** → usually decreases the house price  
- **Higher % lower status population (LSTAT)** → usually decreases the house price  
- **Higher property tax** → usually decreases the house price  
- **Near Charles River** → usually increases the house price
""")

# ---------------------------
# Interactive Bar Chart for Input Features (Foolproof)
# ---------------------------
st.subheader("🔹 Your Input Feature Values")

# Feature Descriptions
feature_names = {
    'rm': "Average number of rooms",
    'age': "Age of the house",
    'crim': "Crime rate",
    'lstat': "% lower status population",
    'tax': "Property tax rate",
    'chas': "Near Charles River"
}

# Ensure all input values are numeric
input_values = input_df.copy()
for col in input_values.columns:
    input_values[col] = pd.to_numeric(input_values[col], errors='coerce')

# Prepare DataFrame for plotting
plot_df = pd.DataFrame({
    'Feature': [feature_names[f] for f in input_values.columns],
    'Value': input_values.iloc[0].values
})

# Plot interactive bar chart
bar_fig = px.bar(
    plot_df,
    x='Feature',
    y='Value',
    labels={'Feature':'Feature', 'Value':'Value'},
    title='Input Feature Values'
)
bar_fig.update_traces(marker_color='teal')
bar_fig.update_layout(yaxis=dict(title='Feature Value'))
st.plotly_chart(bar_fig)


# ---------------------------
# Scatter Plots: Feature vs House Price
# ---------------------------
st.subheader("🔹 How Features Affect House Price")

# Feature Descriptions
# feature_descriptions = {
#     'rm': "Average number of rooms",
#     'lstat': "% lower status population",
#     'crim': "Crime rate in the area",
#     'tax': "Property tax rate",
#     'age': "Age of the house"
# }

features_to_plot = ['rm', 'lstat', 'crim', 'tax', 'age']
for feature in features_to_plot:
    fig = px.scatter(
        df, 
        x=feature, 
        y='medv', 
        trendline="ols",
        title=f'{feature_names[feature]} vs House Price',
        labels={feature: feature_names[feature], 'medv':'House Price ($1000s)'},
        hover_data=[feature,'medv']
    )
    st.plotly_chart(fig)

# ---------------------------
# FAQ Section
# ---------------------------
st.subheader("❓ Frequently Asked Questions (FAQ)")

st.markdown("""
**Q1: How is the house price predicted?**  
- The model looks at key features like number of rooms, age, crime rate, property tax, % lower status population, and proximity to Charles River.  
- It uses a **linear regression** to combine these features into an estimated price.  

**Q2: Will the predicted price always match the actual price?**  
- No. The predicted price is an **estimate**, not a perfect value.  
- Houses with the same features can have different prices due to factors the model doesn’t know (e.g., renovations, view, neighborhood vibe).  

**Q3: What does R² = 0.66 mean?**  
- R² measures how much of the variation in house prices the model can explain.  
- 0.66 means the model explains **66% of the variation** — pretty reasonable for 6 features and a simple linear model.  

**Q4: What is MAE = $3,791?**  
- MAE (Mean Absolute Error) tells us the **average difference between predicted and actual prices**.  
- On average, the model predicts prices within **±$3,791** of the actual price.  

**Q5: Does more rooms always increase price?**  
- Generally yes — more rooms → higher price.  
- But the model combines all features, so one feature alone doesn’t guarantee a specific price change.  

**Q6: Is this model accurate?**  
- The model is **reasonably accurate** for learning and basic predictions.  
- It gives a **good estimate** but is **not perfect** — some predictions will differ from real prices.  
- For professional use, more features and advanced models would be needed. 

**Q7: Can this model be used in a real housing agency?**  
- For learning and demos, yes.  
- For professional pricing decisions, agencies prefer **more features, more advanced models, and lower error**.
""")
