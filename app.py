import streamlit as st

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI Healthcare Assistant Platform")

st.markdown("""
Welcome to the AI Healthcare Platform.

Select a disease detection module from the sidebar.
""")

st.image(
    "http://images.unsplash.com/photo-1581595219315-a187dd40c322",
    use_container_width=True
)

st.success("System Ready")