"""Optional Streamlit dashboard: streamlit run app.py"""
import streamlit as st
import matplotlib.pyplot as plt
from data_generator import generate_line_data
from line_fitting import fit_ols, fit_ransac
from metrics import compare_models
from visualization import plot_line_fitting, plot_comparison_summary
import tempfile, os

st.set_page_config(page_title="RANSAC Lab",page_icon="📈",layout="wide")
st.title("📈 Robust Line Fitting Laboratory")
st.caption("Explore why RANSAC can recover a line when ordinary least squares is distorted by outliers.")
with st.sidebar:
    st.header("Experiment controls")
    n_in=st.slider("Inlier points",20,500,100,10)
    n_out=st.slider("Outlier points",0,300,40,10)
    slope=st.slider("Ground-truth slope",-3.0,5.0,2.0,.1)
    intercept=st.slider("Intercept",-20.0,30.0,5.0,.5)
    noise=st.slider("Gaussian noise (σ)",0.0,5.0,1.0,.1)
    threshold=st.slider("RANSAC distance threshold",.1,10.0,2.5,.1)
    iterations=st.slider("Maximum iterations",50,3000,500,50)
    confidence=st.slider("Target confidence",.80,.999,.99,.001)
    seed=st.number_input("Random seed",min_value=0,max_value=999999,value=42)
if st.button("Run experiment",type="primary",use_container_width=True):
    try:
        ds=generate_line_data(n_in,n_out,slope,intercept,noise,seed=int(seed))
        ols=fit_ols(ds.points)
        ran=fit_ransac(ds.points,threshold,iterations,seed=int(seed),confidence=confidence)
        om,rm=compare_models(ols,ran,ds)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total points",ds.n_points)
        c2.metric("Outlier contamination",f"{ds.outlier_percentage:.1f}%")
        c3.metric("RANSAC consensus",f"{ran.inlier_count}/{ds.n_points}")
        c4.metric("Iterations used",ran.n_iterations)
        st.subheader("Fitted line comparison")
        with tempfile.TemporaryDirectory() as d:
            p1=plot_line_fitting(ds,ols,ran,os.path.join(d,"fit.png"))
            p2=plot_comparison_summary(ds,om,rm,os.path.join(d,"summary.png"))
            st.image(p1,use_container_width=True)
            st.image(p2,use_container_width=True)
        st.subheader("Quantitative evaluation")
        st.dataframe([om.to_dict(),rm.to_dict()],use_container_width=True,hide_index=True)
        st.info(f"OLS: y = {ols.slope:.4f}x + {ols.intercept:.4f}  |  RANSAC: y = {ran.slope:.4f}x + {ran.intercept:.4f}")
        st.caption("Synthetic-data ground truth is used only for evaluation; neither fitting algorithm receives it.")
    except ValueError as e:
        st.error(str(e))
else:
    st.info("Adjust the experiment controls, then select **Run experiment**.")
