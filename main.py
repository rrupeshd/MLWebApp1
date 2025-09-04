# ---- STANDARD LIBS ----
import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# ---- PAGE CONFIG (must be the first Streamlit call) ----
st.set_page_config(page_title="Rupesh Dubey", page_icon=":bar_chart:", layout="wide")

# ---- HELPERS ----
@st.cache_data(show_spinner=False)
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def ensure_state_key(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default

# ---- TOP NAV ----
menuselected = option_menu(
    None,
    ["Home", "Projects", "My Other Web Apps", "About Me"],
    # NOTE: use Bootstrap icon names (no 'bi-' prefix)
    icons=["house", "list-task", "currency-dollar", "person-lines-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

st.markdown("---")

# =========================
# My Other Web Apps
# =========================
if menuselected == "My Other Web Apps":
    ensure_state_key("nav_to_second_app", False)
    ensure_state_key("nav_to_nocode_app", False)

    if st.button("See other Streamlit Apps"):
        st.session_state["nav_to_second_app"] = True

    if st.session_state["nav_to_second_app"]:
        st.markdown("[Open the Income Tax Calculator App](https://incometax.streamlit.app)")
        st.markdown("[Open the SQL Playground App](https://sqlpractice.streamlit.app/)")

    st.write("")  # small spacer

    if st.button("See no-code apps created with Lovable"):
        st.session_state["nav_to_nocode_app"] = True

    if st.session_state["nav_to_nocode_app"]:
        st.markdown("[Open the Rent Receipt App](https://darkreceipt-genius.lovable.app/)")
        st.markdown("[Open the IQ Test App](https://neural-spark-test.lovable.app/)")
        st.markdown("[Open the Excel File Compare App](https://excel-compare.lovable.app/)")  # (fixed duplicate link)

# =========================
# Home
# =========================
if menuselected == "Home":
    lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            st.header(":bar_chart: Welcome To Rupesh Dubey's Web App!")
            st.write("##")
            st.write(
                """
                Hello, welcome to my first web application created using Streamlit in Python. 
                I’ll be posting some of my learnings in Python and ML for Analytics here.
                I’ll keep adding projects on ML and DL frequently.
                
                To contact me, please click on the “About Me” section.
                """
            )
        with right_column:
            if lottie_coding:
                st_lottie(lottie_coding, height=300, key="coding")
            else:
                st.info("Animation failed to load. (Network blocked or URL changed)")

# =========================
# Projects
# =========================
Projectlist = [
    "Gen AI - ChatBot with some personality",
    "Prediction - Linear Regression",          # (fixed spelling)
    "Prediction - RandomForest Regression",    # (fixed spelling)
    "Classification - Decision Tree",
    "Classification System - Multi Algorithms",
]

if menuselected == "Projects":
    Project = st.radio("Select the Project", Projectlist)

    # Lazy imports with friendly error if a module is missing.
    try:
        if Project == "Gen AI - ChatBot with some personality":
            import Project5 as P5
            P5.Pro5()
        elif Project == "Prediction - Linear Regression":
            import Project1 as P1
            P1.Pro1()
        elif Project == "Prediction - RandomForest Regression":
            import Project2 as P2
            P2.Pro2()
        elif Project == "Classification - Decision Tree":
            import Project3 as P3
            P3.Pro3()
        elif Project == "Classification System - Multi Algorithms":
            import Project4 as P4
            P4.Pro4()
    except ModuleNotFoundError as e:
        st.error(f"Required module not found: `{e.name}`. Please add the file to your app folder.")
    except AttributeError as e:
        st.error("The selected project module is missing the expected function. "
                 "Ensure it defines `Pro1/Pro2/Pro3/Pro4/Pro5()` as used here.")
    except Exception as e:
        st.exception(e)

# =========================
# About Me
# =========================
if menuselected == "About Me":
    lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

    col11, col12 = st.columns(2)
    with col11:
        pic = "https://media.licdn.com/dms/image/C4D03AQGcObyFZvfRtQ/profile-displayphoto-shrink_800_800/0/1645786089098?e=2147483647&v=beta&t=o194P0ueezMUp6b7r6e0UGNBWkj8UqXpbN3OvrdZDWY"
        # Some LinkedIn image URLs may expire; wrap in try/except
        try:
            st.image(pic, caption="Me")
        except Exception:
            st.info("Profile image could not be loaded (remote URL blocked or expired).")
    with col12:
        if lottie_hello:
            st_lottie(lottie_hello, speed=1, reverse=False, loop=True, quality="low", key="hello")
        else:
            st.info("Animation failed to load.")

    st.subheader("Certificates")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    cimglink = "https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE"
    with col1:
        st.image(f"{cimglink}~7FLA7JPYU273/CERTIFICATE_LANDING_PAGE~7FLA7JPYU273.jpeg",
                 caption="Python for Data Science, AI & Development")
    with col2:
        st.image(f"{cimglink}~DZSE9773S8A2/CERTIFICATE_LANDING_PAGE~DZSE9773S8A2.jpeg",
                 caption="SQL for Data Science")
    with col3:
        st.image(f"{cimglink}~9CLH6FXWBB3G/CERTIFICATE_LANDING_PAGE~9CLH6FXWBB3G.jpeg",
                 caption="Data Visualization with Tableau")
    with col4:
        st.image(f"{cimglink}~NAJL962VEGM5/CERTIFICATE_LANDING_PAGE~NAJL962VEGM5.jpeg",
                 caption="Basic Statistics")
    with col5:
        st.image(f"{cimglink}~DFU5L2ABS8TD/CERTIFICATE_LANDING_PAGE~DFU5L2ABS8TD.jpeg",
                 caption="Business Metrics for Data-Driven Companies")
    with col6:
        st.image(f"{cimglink}~THW33CM8UBUH/CERTIFICATE_LANDING_PAGE~THW33CM8UBUH.jpeg",
                 caption="Tools for Data Science")

    # ---- Sidebar ----
    with st.sidebar:
        selected = option_menu(
            "Socials",
            ["LinkedIn", "Instagram", "GitHub", "Facebook", "Email"],
            icons=["linkedin", "instagram", "github", "facebook", "envelope"],  # fixed order
            menu_icon="cast",
            default_index=0,
        )

        if selected == "LinkedIn":
            st.markdown("[LinkedIn](https://www.linkedin.com/in/rupeshdubey9/)")
        elif selected == "Instagram":
            st.markdown("[Instagram](https://www.instagram.com/rupeshdubey9/)")
        elif selected == "GitHub":
            st.markdown("[GitHub](https://github.com/rrupeshd)")
        elif selected == "Facebook":
            st.markdown("[Facebook](https://facebook.com/RrupeshD/)")
        elif selected == "Email":
            st.write("Email: rupeshdubey999@gmail.com")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
