
import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie


st.set_page_config(page_title="Rupesh Dubey", page_icon=":bar_chart:", layout="wide")

menuselected = option_menu(None, ["Home", "Projects", 'About Me'],
    icons= ['house', "list-task", 'bi-person-lines-fill'],
    menu_icon="cast", default_index=0, orientation="horizontal")

st.markdown("""---""")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if menuselected == "Home":
    lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
    # ---- WHAT I DO ----
    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            st.header(":bar_chart: Welcome To Rupesh Dubey's - Web App!!!")
            st.write("##")
            st.write(
                """
                Hello, welcome to my first web application created using Streamlit on Python. 
                I will be posting some of my learnings in Python and ML in Analytics field on the go here.
                Will keep adding projects on ML and DL frequently. 
                
                To contact me please click on About Me section.                        
                """
            )
            #st.write("[YouTube Channel >](https://youtube.com/c/CodingIsFun)")
        with right_column:
            st_lottie(lottie_coding, height=300, key="coding")


Projectlist = []

Projectlist.append('Predication - Linear Regression')
Projectlist.append('Predication - RandomForest Regression')
Projectlist.append('Classification - Decision Tree')
Projectlist.append('Classification System - Multi Algorithms')

if menuselected == "Projects":
    Project = st.radio(
        "Select the Project",
        (Projectlist))

    if Project == 'Predication - Linear Regression':
        import Project1 as P1
        P1.Pro1()
    if Project == 'Predication - RandomForest Regression':
        import Project2 as P2
        P2.Pro2()
    if Project == 'Classification - Decision Tree':
        import Project3 as P3
        P3.Pro3()
    if Project == 'Classification System - Multi Algorithms':
        import Project4 as P4
        P4.Pro4()


# Create a button, that when clicked, shows a text
if (menuselected == "About Me"):
    lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")
    col11, col12 = st.columns(2)
    with col11:
        pic="https://scontent-bom1-2.xx.fbcdn.net/v/t1.6435-9/59386096_2260099934046126_3002959138142552064_n.jpg?_nc_cat=109&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=sbBjovrI6tcAX_-ZxL5&_nc_ht=scontent-bom1-2.xx&oh=00_AT9FOOsjnZifoVrc0W5RU4fgmNgrqyni9abEsulni7_6AQ&oe=62D28171"
        st.image(pic, caption="Me", output_format="auto")
    with col12:
        with st.container():
            st_lottie(lottie_hello, speed=1, reverse=False, loop=True,
                    quality="low", key=None)

    st.subheader("Certificates")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    cimglink="https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE"
    with col1:
        image=cimglink+"~7FLA7JPYU273/CERTIFICATE_LANDING_PAGE~7FLA7JPYU273.jpeg"
        st.image(image, caption="Python for Data Science, AI & Development", output_format="auto")
    with col2:
        image=cimglink+"~DZSE9773S8A2/CERTIFICATE_LANDING_PAGE~DZSE9773S8A2.jpeg"
        st.image(image, caption="SQL for Data Science", output_format="auto")
    with col3:
        image=cimglink+"~9CLH6FXWBB3G/CERTIFICATE_LANDING_PAGE~9CLH6FXWBB3G.jpeg"
        st.image(image, caption="Data Visualization and Communication with Tableau", output_format="auto")
    with col4:
        image=cimglink+"~NAJL962VEGM5/CERTIFICATE_LANDING_PAGE~NAJL962VEGM5.jpeg"
        st.image(image, caption="Basic Statistics", output_format="auto")
    with col5:
        image=cimglink+"~DFU5L2ABS8TD/CERTIFICATE_LANDING_PAGE~DFU5L2ABS8TD.jpeg"
        st.image(image, caption="Business Metrics for Data-Driven Companies", output_format="auto")
    with col6:
        image=cimglink+"~THW33CM8UBUH/CERTIFICATE_LANDING_PAGE~THW33CM8UBUH.jpeg"
        st.image(image, caption="Tools for Data Science", output_format="auto")

    # st.balloons()

    with st.sidebar:
        selected = option_menu("Socials", ["LinkedIN", 'Instagram', 'Github','Facebook', 'Email'],
                               icons=['linkedin', 'instagram', 'facebook', 'github', 'envelope'],
                               menu_icon="cast", default_index=1)

        if selected == "LinkedIN":
            link = "[LinkedIN](https://www.linkedin.com/in/rupeshdubey9/)"
        if selected == "Instagram":
            link = "[Instagram](https://www.instagram.com/rupeshdubey9/)"
        if selected == "Github":
            link = "[Github](https://github.com/rrupeshd)"
        if selected == "Facebook":
            link = "[Facebook](https://facebook.com/RrupeshD/)"
        if selected == "Email":
            link = "Email :- rupeshdubey999@gmail.com"
        st.write(link, unsafe_allow_html=True)


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

