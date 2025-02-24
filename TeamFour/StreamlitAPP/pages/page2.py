import streamlit as st

st.header("📈 Page 2 - Insights")
st.write("This is Page 2. Add your content here!")

# Example bar chart
import pandas as pd
import numpy as np
data = pd.DataFrame(np.random.randn(10, 3), columns=['X', 'Y', 'Z'])
st.bar_chart(data)
