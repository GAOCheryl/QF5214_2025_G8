import streamlit as st

st.header("📊 Page 1 - Data Overview")
st.write("This is Page 1. Add your content here!")

# Example chart
import pandas as pd
import numpy as np
data = pd.DataFrame(np.random.randn(20, 3), columns=['A', 'B', 'C'])
st.line_chart(data)
