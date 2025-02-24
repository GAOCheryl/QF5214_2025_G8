import streamlit as st

# Set up page title and layout
st.set_page_config(page_title="Team Four Dashboard", layout="wide")

# Main Content Area (No Sidebar, No Navigation)
st.title("Welcome to Team Four's Dashboard! 🎯")
st.write("This is the main page of the dashboard.")

# You can add any other content here
st.subheader("Dashboard Overview")
st.write("Add charts, tables, or metrics here.")

# Example: Show a sample chart
import pandas as pd
import numpy as np
data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
st.line_chart(data)
