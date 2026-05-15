import streamlit as st
    
def artist_header(artist, image_url, display_name):

    col1, col2 = st.columns([1, 5])

    with col1:
        if image_url:
            st.image(image_url)

            # petit hack visuel : bordure via CSS Streamlit
            st.markdown(
                """
                <style>
                [data-testid="stImageContainer"] img {
                    width: 120px !important;
                    height: 120px !important;
                    border-radius: 50% !important;
                    object-fit: cover !important;
                    border: 4px solid #5dbf8a !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

    with col2:
        st.markdown(
            f"<h1 style='margin:10px 0 0 0'>{display_name}</h1>",
            unsafe_allow_html=True
        )