import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="UniMatch AI", layout="wide")
st.title("🎓  UniMatch AI")
st.markdown("Find your ideal CS/AI university here")

@st.cache_data
def load_data():
    df = pd.read_csv("unis.csv")
    return df

df = load_data()

st.sidebar.header("Your Profile")

gpa = st.sidebar.slider(
    "Your GPA (on 4.0 scale)",
    min_value=2.5,
    max_value=4.0,
    value=3.5,
    step=0.1
)

max_tuition_usd = st.sidebar.number_input(
    "Max Annual Tuition (USD)",
    min_value=0,
    max_value=100000,
    value=40000,
    step=5000
)

location_pref = st.sidebar.multiselect(
    "Preferred Regions",
    options=["Asia", "North America", "Europe", "Australia"],
    default=["Asia", "North America"]
)

interest = st.sidebar.selectbox(
    "Primary Interest",
    options=["AI/ML", "Systems", "Theory", "Robotics", "Entrepreneurship"]
)

region_map = {
    "Singapore": "Asia",
    "USA": "North America",
    "Canada": "North America",
    "Switzerland": "Europe",
    "UK": "Europe",
    "China": "Asia",
    "India": "Asia",
    "Australia": "Australia"
}

df['Region'] = df['Location'].map(region_map)

filtered_df = df[
    (df['Min_GPA'] <= gpa) &
    (df['Max_Tuition'] <= max_tuition_usd) &
    (df['Region'].isin(location_pref))
].copy()

if filtered_df.empty:
    st.warning("No university matches your hard criteria. Try relaxing your filters")
    st.stop()

st.write(f"{len(filtered_df)} universities match your filters") 

filtered_df['Combined_Features'] = filtered_df['Strengths'] + " " + filtered_df["Program_Type"]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(filtered_df['Combined_Features'])

user_vector = vectorizer.transform([interest])

similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

filtered_df["Match_Score"] = similarities * 100

filtered_df = filtered_df.sort_values("Match_Score", ascending=False)

st.subheader("🏆 Your Top University Matches")

st.caption(f"{len(filtered_df)} universities match your criteria. Sorted by AI match score.")

for i, row in filtered_df.head(5).iterrows():
    with st.container():
        # Create two columns: left for details, right for GPA metric
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # University name and location
            st.markdown(f"### {row['Name']}")
            st.markdown(f"📍 {row['Location']} | 💰 Tuition: ${row['Max_Tuition']:,} USD | 📊 Acceptance: {row['Acceptance_Rate']}%")
            
            # Strengths and match score
            st.markdown(f"**Strengths:** {row['Strengths']}")
            st.markdown(f"**Match Score:** {row['Match_Score']:.1f}%")
            
            # Optional: Add a progress bar for visual match score
            st.progress(row['Match_Score'] / 100)
        
        with col2:
            # Display GPA requirement as a metric card
            st.metric("GPA Req", row['Min_GPA'])
        
        # Add a subtle divider between universities
        st.divider()

# Show a note if more than 5 universities matched
if len(filtered_df) > 5:
    st.caption("Showing top 5 matches. Adjust filters to see more.")