
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "manual.pdf"


def display_policy_manual():

    st.title("HR Policy Manual")
    st.caption(
        "Official organizational HR policy document. "
        "Refer to this manual while interacting with the assistant."
    )

    if not PDF_PATH.is_file():
        st.error("HR Policy Manual could not be found.")
        st.code(str(PDF_PATH))
        return

    pdf_bytes = PDF_PATH.read_bytes()

    file_size_mb = len(pdf_bytes) / (1024 * 1024)

    col1, col2 = st.columns(2)

    col1.metric("Document size", f"{file_size_mb:.2f} MB")
    col2.metric("Document type", "PDF")

    st.download_button(
        label="Download HR Policy Manual",
        data=pdf_bytes,
        file_name="HR_Policy_Manual.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.divider()

    st.subheader("Document Viewer")

    # Native Streamlit PDF viewer
    st.pdf(
        pdf_bytes,
        height=850,
    )
