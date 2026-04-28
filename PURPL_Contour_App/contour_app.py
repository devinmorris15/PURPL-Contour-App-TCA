import streamlit as st
import base64
import numpy as np
from rocketcea.cea_obj import CEA_Obj
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import ezdxf
import io
import plotly.graph_objects as go
from itertools import chain
from matplotlib.patches import Arc
from bisect import bisect_left
import pandas as pd
import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Import Rafa Heat Code
import Bartz_Values
#import HeatTransferSolver_2D

#import 1D heat transfer code
import heat_transfer_sim as hts

#import materials json
with open(os.path.join(BASE_DIR, "data/materials_1d.json")) as f:
	mats = json.load(f)

############
#To Do
# - Add Conic Nozzle Code Within Contour Generation
# - Add in output table with nozzle size values 
# - Add in thermal coefficients, specific impulse, etc
# - Add in output unit selection for size and pressure, implement into images
# - Add in Rafa's Heat Code
# - Include CEA Graphs (isp, temp, c*, etc)
############

#PURPL Colors
#Dusk (black) hex is #181618
#moon (white) hex is #ffffff
#Stardust (purpl) hex is #9100FF

########################################
#Conversion Factors
#Conversion factors obtained from https://www.unitconverters.net/
########################################

#Pressure Conversions
psi_to_pa = 6894.7572931783  #psia to pascals conversion
kpa_to_pa = 1000   #kpa to pascals
atm_to_psi = 14.6959487755  #atm to psi
atm_to_pa = 101325  #atm to pascals
bar_to_psi = 14.503773773  #bar to psi
bar_to_pa = 100000  #bar to pascals

#Size Conversions
in_to_m = 0.0254  #inches to meters
ft_to_in = 12  #feet to inches
in_to_cm = 2.54  #inches to centimeters
in_to_mm = 25.4  #inches to millimeters
m_to_cm = 100  #meters to centimeters
m_to_mm = 1000  #meters to millimeters

#Mass Conversions
kg_to_lbm = 2.2046226218  #kg to lbm conversion
lbf_N = 4.4482216153  #lbf to N conversion, N = kg-m/s^2

#Temperature Conversions
rankineToKelvin = 5.0 / 9.0  #Rankine to Kelvin conversion
rankineToF = -459.67  #Rankine to Fahrenheit conversion
celciusToKelvin = 273.15

#heat value conversions
cp_conversion = 4186.8
tc_conversion = 418.4
visc_conversion = 0.0001

########################################
#Global Constants
########################################
g = 9.81  #gravity in m/s^2
mach_ms = 343  #Mach speed in m/s
R = 8314.462618 #universal gas constant in J/kg-K
########################################

st.set_page_config(page_title="PURPL Rocket Nozzle Contouring", layout="wide")

#Global Font Customization
st.markdown("""
	<style>
		@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=Roboto+Mono:wght@400;700&display=swap');

		/* Headers — Chakra Petch */
		h1, h2, h3, h4, h5, h6,
		.stTitle, .stHeader, .stSubheader {
			font-family: 'Chakra Petch', sans-serif !important;
		}

		/* Regular text — Roboto Mono normal */
		html, body, p, div, span, label,
		.stMarkdown, .stText, .stSelectbox,
		.stNumberInput, [data-testid="stWidgetLabel"] {
			font-family: 'Roboto Mono', monospace !important;
			font-weight: 400 !important;
		}

		/* Caption/label text — Roboto Mono bold */
		.stCaption, small,
		[data-testid="stSidebar"] label,
		[data-testid="stSidebar"] .stSelectbox label,
		[data-testid="stSidebar"] .stNumberInput label {
			font-family: 'Roboto Mono', monospace !important;
			font-weight: 700 !important;
		}

		/* Tabs — Chakra Petch to match headers */
		.stTabs [data-baseweb="tab"] {
			font-family: 'Chakra Petch', sans-serif !important;
			font-weight: 600 !important;
		}
	</style>
""", unsafe_allow_html=True)

st.markdown("""
	<style>
		[data-testid="collapsedControl"],
		[data-testid="stSidebarCollapseButton"] {
			display: none !important;
		}
	</style>
""", unsafe_allow_html=True)

#Tabs customization (white unselected, purpl when active or hovered over)
st.markdown("""
    <style>
        /* Inactive tabs — white */
        .stTabs [data-baseweb="tab"] {
            color: #ffffff !important;
            font-family: 'Roboto Mono', monospace !important;
            font-weight: 400 !important;
            font-size: 1.1rem !important;
            padding: 10px 16px !important;
            margin-right: 8px !important;
        }
        /* Active/selected tab — purple */
        .stTabs [aria-selected="true"] {
            color: #9100FF !important;
            font-family: 'Roboto Mono', monospace !important;
            font-weight: 400 !important;
        }
        /* Hovered tab — purple */
        .stTabs [data-baseweb="tab"]:hover {
            color: #9100FF !important;
        }
        /* Tab underline indicator */
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #9100FF !important;
        }
		/* Target the inner text span directly */
		.stTabs [data-baseweb="tab"] p {
			font-size: 1.1rem !important;
			margin: 0 !important;
		}
    </style>
""", unsafe_allow_html=True)

#Font exclusion for the <</>> button within the sidebar due to bug
st.markdown("""
	<style>
		/* Reset all sidebar toggle arrow buttons */
		[data-testid="collapsedControl"],
		[data-testid="baseButton-headerNoPadding"],
		[data-testid="stSidebarCollapseButton"],
		[data-testid="stSidebarExpandButton"],
		button[kind="header"] {
			font-family: initial !important;
		}
		
		/* Reset the icon spans inside those buttons */
		[data-testid="stSidebarCollapseButton"] span,
		[data-testid="stSidebarCollapseButton"] p,
		[data-testid="stSidebarExpandButton"] span,
		[data-testid="stSidebarExpandButton"] p,
		[data-testid="collapsedControl"] span,
		[data-testid="collapsedControl"] p {
			font-family: 'Material Symbols Rounded', sans-serif !important;
			font-size: 20px !important;
		}
	</style>
""", unsafe_allow_html=True)

st.markdown("""
	<style>
		@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=Roboto+Mono:wght@400;700&display=swap');

		/* ... all your existing styles ... */

		/* Image background fix */
		[data-testid="stImage"],
		[data-testid="stImage"] img,
		[data-testid="stImage"] > div {
			background-color: transparent !important;
			background: transparent !important;
		}
	</style>
""", unsafe_allow_html=True)

#Imports PURPL Logo
def load_logo(path):
	with open(path, "rb") as f:
		return base64.b64encode(f.read()).decode()
logo_b64 = load_logo(os.path.join(BASE_DIR, "assets/purpl_transparent_logo.png"))

#load my personal picture
def get_image_base64(path):
    abs_path = os.path.join(os.path.dirname(__file__), path)
    with open(abs_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

devin_pic = get_image_base64("assets/devin_purpl_pic.jpeg")

#Adds App Header With PURPL Logo and Title, gotten from Claude
st.markdown(f"""
	<style>
		.stAppViewContainer {{
			margin-top: 85px;
		}}
		.custom-header {{
			position: fixed;
			top: 0; left: 0; right: 0;
			height: 80px;                /* ← increased from 65px */
			background-color: #1a1a2e;
			border-bottom: 2px solid #9100FF;
			display: flex;
			align-items: center;
			padding: 0 24px;
			gap: 16px;
			z-index: 9999;
		}}
		.custom-header img {{
			height: 52px;               /* ← slightly bigger logo to match */
		}}
		.custom-header h1 {{
			color: #9100FF;
			font-size: 1.8rem;          /* ← slightly bigger text to fill space */
			margin: 0;
			line-height: 1;
		}}
	</style>

	<div class="custom-header">
		<img src="data:image/png;base64,{logo_b64}"/>
		<div>
			<h1>Purdue Undergraduate Rocket Propulsion Laboratory - Nozzle Contouring App</h1>
		</div>
	</div>
""", unsafe_allow_html=True)

#sticky tabs
st.markdown("""
    <style>
        /* Sticky tabs */
        .stTabs [data-baseweb="tab-list"] {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: #0e1117;
            padding-top: 8px;
        }
    </style>
""", unsafe_allow_html=True)

#Hides dark mode option selection
st.markdown("""
    <style>
        /* Hide the settings/theme toggle button */
        #MainMenu {visibility: hidden;}
        header[data-testid="stHeader"] button[kind="header"] {display: none;}
    </style>
""", unsafe_allow_html=True)

#remove blank space at the top of app
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
            margin-top: -7rem;
        }
    </style>
""", unsafe_allow_html=True)

#removes press enter to apply pop up
st.markdown("""
    <style>
		/* Hide 'Press Enter to apply' tooltip */
        [data-testid="InputInstructions"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

#remove blank space at top of sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] .block-container {
            padding-top: 0rem;
            margin-top: -6rem;
        }
        section[data-testid="stSidebar"] > div {
    		padding-top: 0rem;
    		margin-top: -3rem;  /* go as negative as needed */
		}
    </style>
""", unsafe_allow_html=True)

#create sidebar
with st.sidebar:

	st.title("Input Parameters")

	#Propellant Choices
	st.subheader("Propellant Choices")
	#Make standard selection and full list
	prop_choices = st.selectbox("Propellant List", ["Simple Propellants List", "Full Propellants List"], index = 0)

	col1, col2, col3 = st.columns([1.5, 1.5, 1])

	if prop_choices == "Simple Propellants List":
		with col1:
			ox = st.selectbox("Oxidizer", ["LOX", "N2O"], index=0) #Choose Oxidizer Select Box
		with col2:
			fuel = st.selectbox("Fuel", ["RP_1", "LH2", "CH4", "Isopropanol"], index=0) #Choose Fuel Select Box
	else:
		with col1:
			ox = st.selectbox("Oxidizer", ["90_H2O2", "98_H2O2","LOX", "N2O"], index=0) #Choose Oxidizer Select Box
		with col2:
			fuel = st.selectbox("Fuel", ["CH4", "Ethanol", "Gasoline", "HTBP", "Isopropanol", "Kerosene", "LCH4_NASA", "LH2_NASA", "Methanol", "MMH", "N2H4", "Propane", "RP_1", "RP1_NASA"], index=0) #Choose Fuel Select Box
    #a
	with col3:
		of_ratio = st.number_input(label="O/F Ratio", min_value=0.0, max_value=100.0, value=None, step=0.1)#Input Box for O/F Ratio

	##########################################################################
	#Frozen propellents

	#Are propellants frozen
	col1, col2 = st.columns(2)

	with col1:
		frozen = st.selectbox("Propellant State", ["Equilibrium", "Frozen"], index=0) #Select if propellants are frozen
	with col2:
		freeze_location = st.selectbox("Location of Freeze", ["----", "Chamber", "Throat"], index=0, disabled=frozen not in ["Frozen"])
	if frozen == "Equilibrium":
		freeze_location = None

	st.divider()
	##########################################################################
	#Thrust Inputs

	st.subheader("Thrust Characterization")

	thrust_def = st.radio("Define thrust by:", options=["Force of Thrust", "Mass Flow Rate"], horizontal=True, key="thrust_defintion")

	if thrust_def == "Force of Thrust":
		col1, col2 = st.columns([2,1])
		with col1:
			thrust = st.number_input(label="Thrust", min_value=0.0, max_value=10000000.0, value=None, step=10.0) #Input Box for Thrust
		with col2: 
			thrust_unit = st.selectbox("Units", ["lbf", "N"], index=0, key = "thrust__unit") #Thrust Unit Select Box
	elif thrust_def == "Mass Flow Rate":
		col1, col2 = st.columns([2,1])
		with col1:
			mass_flow = st.number_input(label="Mass Flow Rate", min_value=0.0, max_value=10000000.0, value=None, step=10.0) #Input Box for Thrust
		with col2: 
			mass_flow_unit = st.selectbox("Units", ["lb/s", "kg/s"], index=0, key = "mdot__unit") #Thrust Unit Select Box

	col1, col2 = st.columns(2)
	with col1:
		characteristic_velo_eff = st.number_input(label="Characteristic Velocity (c*) Efficiency %", min_value=0.0, max_value=100.0, value=100.0) #Input Box for c* efficiency
		#ef_cstar = characteristic_velo_eff / 100.0
	with col2:
		coefficient_thrust_eff = st.number_input(label="Coefficient of Thrust (cf) Efficiency %", min_value=0.0, max_value=100.0, value=100.0) #Input Box for c* efficiency
		#ef_cf =  coefficient_thrust_eff / 100.0

	st.divider()
	##########################################################################
	#Pressures
	st.subheader("Pressures")

	col1, col2 = st.columns([2,1])
	#Chamber Pressure Inputs
	with col1:
		chamber_pressure = st.number_input(label="Chamber Pressure", min_value=0.0, max_value=10000000.0, value=None, step=10.0)#Input Box for Chamber Pressure
	with col2:
		chamber_pressure_unit = st.selectbox("Units", ["psia", "Pa", "kPa","atm", "bar"], index=0, key = "pc_unit") #Chamber Pressure Unit Select Box

	#Exit Pressure Inputs
	col1, col2 = st.columns([2,1])
	with col1:
		exit_pressure = st.number_input(label="Exit Pressure", min_value=0.0, max_value=10000000.0, value=None, step=10.0)#Input Box for Exit Pressure
	with col2:
		exit_pressure_unit = st.selectbox("Units", ["psia", "Pa", "kPa","atm", "bar"], index=0, key = "pe_unit") #Exit Pressure Unit Select Box

	#Ambient Pressure Inputs
	col1, col2 = st.columns([2,1])
	with col1:
		ambient_pressure = st.number_input(label="Ambient Pressure", min_value=0.0, max_value=10000000.0, value=None, step=10.0)#Input Box for Ambient Pressure
	with col2:
		ambient_pressure_unit = st.selectbox("Units", ["psia", "Pa", "kPa","atm", "bar"], index=0, key = "pamb_unit") #Ambient Pressure Unit Select Box


	st.divider()
	##########################################################################
	#Overall Geometry

	#Chamber Geometry
	st.subheader("Chamber Geometry")
	
	chamber_size_def = st.radio("Define chamber geometry by:", options=["Contraction Ratio", "Chamber Diameter"], horizontal=True, key="chamb_size_selection")

	if chamber_size_def == "Contraction Ratio":
		con_ratio = st.number_input("Contraction Ratio", min_value=1.0, value=None)
		chamber_dia = None
		chamber_dia_unit = None
	else:
		con_ratio = None
		col1, col2 = st.columns([2,1])
		with col1:
			chamber_dia = st.number_input("Chamber Diameter", min_value=0.0, value=None, format="%.3f")
		with col2:
			chamber_dia_unit = st.selectbox("Units", ["in", "mm", "cm", "m"], index=0, key = "dc_unit") #Unit for chamber diameter selection

	contraction_angle = st.number_input(label="Contraction Angle (Degrees)", min_value=0.0, max_value=90.0, value=None, step=1.0)#Input Box for Contraction Angle

	chamber_len_def = st.radio("Define Chamber Length By:", options=["Characteristic Length (L*)", "Chamber Length (Lc)"], horizontal=True, key="length_selection")

	col1, col2 = st.columns([2,1])
	with col1:
		if chamber_len_def == "Characteristic Length (L*)":
			characteristic_length = st.number_input(label="Characteristic Length (L*)", min_value=0.0, max_value=10000.0, value=None, step=1.0)#Input Box for Charactersitic Length
		else:
			chamber_length = st.number_input(label="Chamber Length (Lc)", min_value=0.0, max_value=10000.0, value=None, step=1.0)#Input Box for Chamber Length
	with col2:
		if chamber_len_def == "Characteristic Length (L*)":
			characteristic_length_unit = st.selectbox("Units", ["in", "mm", "cm", "m"], index=0, key = "lstar_unit") #Unit for characteristic length selection
			chamber_length_unit = None
		else:
			chamber_length_unit = st.selectbox("Units", ["in", "mm", "cm", "m"], index=0, key = "lc_unit") #Unit for chamber length selection
			characteristic_length_unit = None
	#Nozzle Geometry

	st.subheader("Nozzle Geometry")

	col1, col2 = st.columns(2)
	with col1:
		nozzle_type = st.selectbox("Nozzle Type", ["Bell Nozzle", "Conic Nozzle"])
	with col2:
		if nozzle_type == "Conic Nozzle":
			bell_percent = None
			divergent_angle = st.number_input(label="Divergent Angle (Degrees)", min_value=0.0, max_value=90.0, value=None, step=1.0)
		else:
			bell_percent = st.number_input("Bell Percent Length", min_value=60.0, max_value = 100.0, value=None, step = 10.0)
			divergent_angle = None

	#Expansion Ratio Choice If Wanted

	#Curve Radii
	col1, col2, col3 = st.columns(3)
	with col1:
		radius1 = st.number_input("R_1/R_t", min_value=0.0, value=1.5, format="%.3f")
	with col2:
		radius2 = st.number_input("R_2/R_2max", min_value=0.0, max_value = 1.0, value=0.3, format="%.3f")
	with col3:
		radiusn = st.number_input("R_n/R_t", min_value=0.0, value=0.382, format="%.3f") 

	st.divider()

	st.subheader("Output Options")

	twod_contour = st.checkbox("Generate 2D Contour", value = True)
	threed_contour = st.checkbox("Generate 3D Contour", value = False)
	csv_plot = st.checkbox("Generate Contour Plot CSV File", value = False)
	dxf_plots = st.checkbox("Generate Contour Plot DXF File", value = False)
	out_len_unit = st.radio("Choose Length Output Units:", options=["in" , "mm", "cm", "m"], horizontal=True, key="out_len_unit")
	out_p_unit = st.radio("Choose Pressure Output Units:", options=["psia", "Pa", "kPa", "atm", "bar"], horizontal=True, key="out_p_unit")
	out_m_unit = st.radio("Choose Mass Output Units:", options=["lb", "kg"], horizontal=True, key="out_m_unit")
	
	st.divider()

	# 1. CSS styling (put this near the top with your other styles)
	st.markdown("""
		<style>
			.stButton > button {
				background-color: #9100FF;
				color: #ffffff;
				border: none;
			}
			.stButton > button:hover {
				background-color: #9100FF;
				color: #ffffff;
				border: 2px solid #ffffff;
			}
			.stButton > button:active {
				background-color: #ffffff;
				color: #9100FF;
				border: none;
			}
		</style>
	""", unsafe_allow_html=True)

	run_contour = st.button("Generate Contour", type="primary", use_container_width=True, key="run_cont_btn")

#tab1, tab2, tab3, tab4 = st.tabs(["overview", "nozzle contour results", "other results", "1-D Thermal Sims"])
tab1, tab2, tab3 = st.tabs(["overview", "nozzle contour results", "more results"])

with tab1:
    st.title("Welcome to PURPLContour!")
    st.divider()
    st.markdown(f"""
        <p>PURPLContour is a free app designed to be used by PURPL and other student design teams around the world to improve accessibility and participation for rocketry.</p>

        <div style="overflow: hidden; border: 1px solid #DBC885; border-radius: 8px; padding: 16px;">
            <div style="float: left; margin-right: 24px; margin-bottom: 8px; text-align: center;">
                <img src="data:image/jpeg;base64,{devin_pic}" width="180" style="border-radius: 8px; display: block;"/>
                <div style="margin-top: 8px; font-family: 'Roboto Mono', monospace; font-size: 0.85rem; color: #cccccc; line-height: 1.6;">
                    <strong>Devin Morris</strong><br>
                    Mechanical Engineering '28<br>
                    Discord: @yungstarfish.
                </div>
            </div>
            The Thrust Chamber Assembly subteam within PURPL's Turbopump team developed PURPLContour as an alternative to paywalled team standards. The app is designed to allow diverse rocket engine designs with unique specifications to obtain contour designs. Both 2D and 3D graphics of the contour may be obtained, as well as both CSV and DXF outputs for implementation flexibility with numerous CAD and simulation softwares.
            <br><br>
            Our team is dedicated to delivering a continuously improving interface and accurate results. Contact our app's creator, Devin, on discord to provide feedback!
        </div>
    """, unsafe_allow_html=True)

#	st.text("We are Purdue Undergraduate Rocket Propulsion Laboratory, a forward-thinking team of undergraduate students designing, building, and testing rocket engines.")

with tab3:
	 st.title("More Results")

with tab2:
	st.title("Nozzle Contour Results")
	st.divider()

#with tab4:
#	st.title("1-Dimension Heat Transfer Simulator")

######################################
#User-Defined Functions
######################################

def pressure_convert(variable, var_unit):    #This function takes in the pressure input by user and ouputs the pressure in both psia and pascals
	if var_unit == "psia":
		var_psi = variable
		var_pa = variable * psi_to_pa
	elif var_unit == "kPa":
		var_psi = variable * kpa_to_pa / psi_to_pa
		var_pa = variable * kpa_to_pa
	elif var_unit == "Pa":
		var_psi = variable / psi_to_pa
		var_pa = variable
	elif var_unit == "bar":
		var_psi = variable * bar_to_psi
		var_pa = variable * bar_to_pa
	elif var_unit == "atm":
		var_psi = variable * atm_to_psi
		var_pa = variable * atm_to_pa

	return var_psi, var_pa

def out_pressure(variable, var_unit):
		if var_unit == "psia":
			var_out = variable / psi_to_pa
		elif var_unit == "kPa":
			var_out = variable / kpa_to_pa
		elif var_unit == "Pa":
			var_out = variable
		elif var_unit == "bar":
			var_out = variable / bar_to_pa
		elif var_unit == "atm":
			var_out = variable / atm_to_pa
		return(var_out)

def conv_to_m(variable, var_unit):
	if var_unit == "in":
		conv_var = variable * in_to_m
	elif var_unit == 'mm':
		conv_var = variable / m_to_mm
	elif var_unit == 'm':
		conv_var = variable
	else:
		conv_var = variable

	return(conv_var)

def te_sizing(thrust, mflow, mr, pc, pe, pamb, ef_cstar, ef_cf, fr1, fr2):
	Eps = C.get_eps_at_PcOvPe(Pc = pc, MR = mr, PcOvPe= (pc / pe), frozen=fr1, frozenAtThroat=fr2)      #Calculates optimal expansion ratio
	Cstar = C.get_Cstar(Pc = pc, MR = mr) * in_to_m * ft_to_in * ef_cstar          #Calculates characteristic velocity in m/s, with efficiency factor
	cf_arr = C.get_PambCf(Pamb = pamb, Pc = pc, MR = mr, eps = Eps)     
	cf = cf_arr[0] * ef_cf                                                 #Calculates coefficient of thrust, with efficiency factor
	Me = C.get_MachNumber(Pc=pc, MR=mr, eps=Eps, frozen=fr1, frozenAtThroat=fr2)
	MW_e, gamma_e = C.get_exit_MolWt_gamma(Pc=pc, MR=mr, eps=Eps, frozen=fr1, frozenAtThroat=fr2)
	Tc, Tt, Te = C.get_Temperatures(Pc=pc, MR=mr, eps=Eps, frozen=fr1, frozenAtThroat=fr2)
	Te_K = Te * rankineToKelvin
	Ve = Me * np.sqrt((gamma_e * R * Te_K) / MW_e)

	if thrust == 0:
		mdot = mflow
		At = (mdot * Cstar) / (pc * psi_to_pa)           #Calculates area of throat in m^2
		Ae = At * Eps                    #Calculates exit area in m^2
		Ft = ((mdot * Ve * (ef_cstar * ef_cf)) + (pe - pamb) * Ae) #adjusted for efficiency factors
		#Ft = (mdot) * (Cstar * ef_cstar) * (cf * ef_cf)
	else:
		Ft = thrust          #Converts force of thrust to Newtons
		At = Ft / (pc * psi_to_pa * cf)
		Ae = At * Eps                    #Calculates exit area in m^2
		mdot = (Ft - ((pe - pamb) * Ae)) / Ve

	#ADD Extra calculation for mass flow rate 

	#At_in = At * ((1 / in_to_m) ** 2)             #Converts area of throat to in^2
	#At_cm = At * (mcm ** 2)                #Converts area of throat to cm^2
	#Dt_in = 2 * np.sqrt(At_in / np.pi)     #Calculates throat diameter in inches
	#De_in = 2 * np.sqrt(Ae / np.pi)     #Calculates exit diameter in inches

	return (Ft, mdot, At, Eps, Ae)

def c_sizing(throat_A, chamber_D, con_r, Lstar, Lc, con_angle):
	if chamber_D == 0:
		Dc = 2.0 * np.sqrt((throat_A * con_r) / np.pi)
		cr = con_r
	else:
		cr = (np.pi * (chamber_D / 2.0) ** 2) / throat_A
		Dc = chamber_D
	if Lc == 0:
		Lc = (Lstar - (1.0/3.0) * np.sqrt(throat_A / np.pi) * (1 / np.tan(np.deg2rad(con_angle))) * (cr **(1.0/3.0) - 1)) / cr
	return(Dc, cr, Lc)

#################################
#Countour Generation Code

# sp.heat, area_ratio, throat_radius, length percentage, 
def bell_nozzle(aratio, Rt, l_percent, cratio, alpha, Lc, r1t, r2m, rnt):
	# upto the nozzle designer, usually -135
	entrant_angle  	= -90 - alpha
	ea_radian 		= math.radians(entrant_angle)
	Rc = Rt * np.sqrt(cratio)

	# nozzle length percntage
	if l_percent == 60:		Lnp = 0.6
	elif l_percent == 80:	Lnp = 0.8
	elif l_percent == 90:	Lnp = 0.9	
	else:					Lnp = 0.8
	# find wall angles (theta_n, theta_e) for given aratio (ar)		
	angles = find_wall_angles(aratio, throat_radius, l_percent)
	# wall angles
	nozzle_length = angles[0]; theta_n = angles[1]; theta_e = angles[2];

	data_intervel  	= 100
	# entrant functions
	ea_start 		= ea_radian
	ea_end 			= -math.pi/2	
	angle_list 		= np.linspace(ea_start, ea_end, data_intervel)
	xe = []; ye = [];
	for i in angle_list:
		xe.append( r1t * Rt * math.cos(i) )
		ye.append( r1t * Rt * math.sin(i) + (1 + r1t) * Rt )

	# linear convergent section functions
	R2max = (Rc - ye[0]) / (1 - np.cos(math.radians(alpha)))
	R2 = r2m * R2max
	diag_x0		= xe[0]
	diag_y0 	= ye[0]	
	diag_yf = Rc - R2 * (1 - np.cos(math.radians(alpha)))
	diag_xf = (diag_yf - diag_y0) / (-np.tan(np.radians(alpha))) + diag_x0
	iters 		= np.linspace(diag_x0, diag_xf, data_intervel)
	xed = []; yed = [];
	for i in iters:
		xed.append( i )
		yed.append( diag_y0 + (abs(i - diag_x0) * np.tan(np.radians(alpha))))

	# Combustion chamber arc functions
	cca_start 		= math.radians(90 - alpha)
	cca_end 		= math.pi/2
	angle_list 		= np.linspace(cca_start, cca_end, data_intervel)
	xeca = []; yeca = [];
	for i in angle_list:
		xeca.append( diag_xf + R2 * math.cos(i) - R2 * math.cos(cca_start))
		yeca.append( diag_yf + R2 * math.sin(i) - R2 * math.sin(cca_start))

	# combustion chamber cylinder functions
	iters 		= np.linspace(xeca[-1], -Lc, data_intervel)
	xecc = []; yecc = [];
	for i in iters:
		xecc.append( i )
		yecc.append( yeca[-1] )

	#exit section
	ea_start 		= -math.pi/2
	ea_end 			= theta_n - math.pi/2
	angle_list 		= np.linspace(ea_start, ea_end, data_intervel)	
	xe2 = []; ye2 = [];
	for i in angle_list:
		xe2.append( rnt * Rt * math.cos(i) )
		ye2.append( rnt * Rt * math.sin(i) + (1 + rnt) * Rt )

	# bell section
	# Nx, Ny-N is defined by [Eqn. 5] setting the angle to (θn – 90)
	Nx = rnt * Rt * math.cos(theta_n - math.pi/2)
	Ny = rnt * Rt * math.sin(theta_n - math.pi/2) + (1 + rnt) * Rt 
	# Ex - [Eqn. 3], and coordinate Ey - [Eqn. 2]
	Ex = Lnp * ( (math.sqrt(aratio) - 1) * Rt )/ math.tan(math.radians(15) )
	Ey = math.sqrt(aratio) * Rt 
	# gradient m1,m2 - [Eqn. 8]
	m1 = math.tan(theta_n);  m2 = math.tan(theta_e);
	# intercept - [Eqn. 9]
	C1 = Ny - m1*Nx;  C2 = Ey - m2*Ex;
	# intersection of these two lines (at point Q)-[Eqn.10]
	Qx = (C2 - C1)/(m1 - m2)
	Qy = (m1*C2 - m2*C1)/(m1 - m2)	
	
	# Selecting equally spaced divisions between 0 and 1 produces 
	# the points described earlier in the graphical method
	# The bell is a quadratic Bézier curve, which has equations:
	# x(t) = (1 − t)^2 * Nx + 2(1 − t)t * Qx + t^2 * Ex, 0≤t≤1
	# y(t) = (1 − t)^2 * Ny + 2(1 − t)t * Qy + t^2 * Ey, 0≤t≤1 [Eqn. 6]		
	int_list = np.linspace(0, 1, data_intervel)
	xbell = [(xe2[-1])]; ybell = [(ye2[-1])];
	for t in int_list:		
		xbell.append( ((1-t)**2)*Nx + 2*(1-t)*t*Qx + (t**2)*Ex )
		ybell.append( ((1-t)**2)*Ny + 2*(1-t)*t*Qy + (t**2)*Ey )
	
	# create negative values for the other half of nozzle
	nye 	= [ -y for y in ye]
	nye2  	= [ -y for y in ye2]
	nyed    = [ -y for y in yed]
	nyeca   = [ -y for y in yeca]
	nyecc   = [ -y for y in yecc]
	nybell  = [ -y for y in ybell]

	# return
	return angles, (xe, ye, nye, xe2, ye2, nye2, xed, yed, nyed, xeca, yeca, nyeca, xecc, yecc, nyecc, xbell, ybell, nybell), R2, nozzle_length

# sp.heat, area_ratio, throat_radius, length percentage, 
def conic_nozzle(aratio, eangle, Rt, cratio, alpha, Lc, r1t, r2m, rnt):
	# upto the nozzle designer, usually -135
	entrant_angle  	= -90 - alpha
	ea_radian 		= math.radians(entrant_angle)
	Rc = Rt * np.sqrt(cratio)
	Re = Rt * np.sqrt(aratio)
	
	#angles = find_wall_angles(aratio, throat_radius, l_percent)
	# wall angles
	#nozzle_length = angles[0]; theta_n = angles[1]; theta_e = angles[2];

	theta_n = eangle * (np.pi / 180)

	data_intervel  	= 100
	# entrant functions
	ea_start 		= ea_radian
	ea_end 			= -math.pi/2	
	angle_list 		= np.linspace(ea_start, ea_end, data_intervel)
	xe = []; ye = [];
	for i in angle_list:
		xe.append( r1t * Rt * math.cos(i) )
		ye.append( r1t * Rt * math.sin(i) + (1 + r1t) * Rt )

	# linear convergent section functions
	R2max = (Rc - ye[0]) / (1 - np.cos(math.radians(alpha)))
	R2 = r2m * R2max
	diag_x0		= xe[0]
	diag_y0 	= ye[0]	
	diag_yf = Rc - R2 * (1 - np.cos(math.radians(alpha)))
	diag_xf = (diag_yf - diag_y0) / (-np.tan(np.radians(alpha))) + diag_x0
	iters 		= np.linspace(diag_x0, diag_xf, data_intervel)
	xed = []; yed = [];
	for i in iters:
		xed.append( i )
		yed.append( diag_y0 + (abs(i - diag_x0) * np.tan(np.radians(alpha))))

	# Combustion chamber arc functions
	cca_start 		= math.radians(90 - alpha)
	cca_end 		= math.pi/2
	angle_list 		= np.linspace(cca_start, cca_end, data_intervel)
	xeca = []; yeca = [];
	for i in angle_list:
		xeca.append( diag_xf + R2 * math.cos(i) - R2 * math.cos(cca_start))
		yeca.append( diag_yf + R2 * math.sin(i) - R2 * math.sin(cca_start))

	# combustion chamber cylinder functions
	iters 		= np.linspace(xeca[-1], -Lc, data_intervel)
	xecc = []; yecc = [];
	for i in iters:
		xecc.append( i )
		yecc.append( yeca[-1] )

	#exit section
	ea_start 		= -math.pi/2
	ea_end 			= theta_n - math.pi/2
	angle_list 		= np.linspace(ea_start, ea_end, data_intervel)	
	xe2 = []; ye2 = [];
	for i in angle_list:
		xe2.append( rnt * Rt * math.cos(i) )
		ye2.append( rnt * Rt * math.sin(i) + (1 + rnt) * Rt )

	# divergent section

	diag_xc		= xe2[-1]
	diag_yc 	= ye2[-1]	
	diag_ycf = Re
	diag_xcf = (diag_ycf - diag_yc) / (np.tan(np.radians(eangle))) + diag_xc
	iters 		= np.linspace(diag_xc, diag_xcf, data_intervel)
	xec = []; yec = [];
	for i in iters:
		xec.append( i )
		yec.append( diag_yc + ((i - diag_xc) * np.tan(np.radians(eangle))))

	nozzle_length = diag_xcf
	
	# create negative values for the other half of nozzle
	nye 	= [ -y for y in ye]
	nye2  	= [ -y for y in ye2]
	nyed    = [ -y for y in yed]
	nyeca   = [ -y for y in yeca]
	nyecc   = [ -y for y in yecc]
	nyec  = [ -y for y in yec]

	# return
	return (xe, ye, nye, xe2, ye2, nye2, xed, yed, nyed, xeca, yeca, nyeca, xecc, yecc, nyecc, xec, yec, nyec), R2, nozzle_length

# find wall angles (theta_n, theta_e) in radians for given aratio (ar)
def find_wall_angles(ar, Rt, l_percent):
	# wall-angle empirical data
	aratio 		= [ 4,    5,    10,   20,   30,   40,   50,   100]
	theta_n_60 	= [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 35.0, 40.0]	
	theta_n_80 	= [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
	theta_n_90 	= [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0]
	theta_e_60 	= [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2]
	theta_e_80 	= [14.0, 13.0, 11.0,  9.0,  8.5,  8.0,  7.5,  7.0]
	theta_e_90 	= [11.5, 10.5,  8.0,  7.0,  6.5,  6.0,  6.0,  6.0]	

	# nozzle length
	f1 = ( (math.sqrt(ar) - 1) * Rt )/ math.tan(math.radians(15) )
	
	if l_percent == 60:
		theta_n = theta_n_60; theta_e = theta_e_60;
		Ln = 0.6 * f1
	elif l_percent == 80:
		theta_n = theta_n_80; theta_e = theta_e_80;
		Ln = 0.8 * f1		
	elif l_percent == 90:
		theta_n = theta_n_90; theta_e = theta_e_90;	
		Ln = 0.9 * f1	
	else:
		theta_n = theta_n_80; theta_e = theta_e_80;		
		Ln = 0.8 * f1

	# find the nearest ar index in the aratio list
	x_index, x_val = find_nearest(aratio, ar)
	# if the value at the index is close to input, return it
	if round(aratio[x_index], 1) == round(ar, 1):
		return Ln, math.radians(theta_n[x_index]), math.radians(theta_e[x_index])

	# check where the index lies, and slice accordingly
	if (x_index>2):
		# slice couple of middle values for interpolation
		ar_slice = aratio[x_index-2:x_index+2]		
		tn_slice = theta_n[x_index-2:x_index+2]
		te_slice = theta_e[x_index-2:x_index+2]
		# find the tn_val for given ar
		tn_val = interpolate(ar_slice, tn_slice, ar)	
		te_val = interpolate(ar_slice, te_slice, ar)	
	elif( (len(aratio)-x_index) <= 1):
		# slice couple of values initial for interpolation
		ar_slice = aratio[x_index-2:len(x_index)]		
		tn_slice = theta_n[x_index-2:len(x_index)]
		te_slice = theta_e[x_index-2:len(x_index)]
		# find the tn_val for given ar
		tn_val = interpolate(ar_slice, tn_slice, ar)	
		te_val = interpolate(ar_slice, te_slice, ar)	
	else:
		# slice couple of end values for interpolation
		ar_slice = aratio[0:x_index+2]		
		tn_slice = theta_n[0:x_index+2]
		te_slice = theta_e[0:x_index+2]
		# find the tn_val for given ar
		tn_val = interpolate(ar_slice, tn_slice, ar)	
		te_val = interpolate(ar_slice, te_slice, ar)						

	return Ln, math.radians(tn_val), math.radians(te_val)

# simple linear interpolation
def interpolate(x_list, y_list, x):
	if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
		raise ValueError("x_list must be in strictly ascending order!")
	intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
	slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

	if x <= x_list[0]:
		return y_list[0]
	elif x >= x_list[-1]:
		return y_list[-1]
	else:
		i = bisect_left(x_list, x) - 1
		return y_list[i] + slopes[i] * (x - x_list[i])

# find the nearest index in the list for the given value
def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx, array[idx]  

# nozzle contour plot
def plot_nozzle(ax, title, Rt, angles, contour, r1t, r2m, rnt, nozz_len):
	# wall angles
	nozzle_length = angles[0]; theta_n = angles[1]; theta_e = angles[2];

	# contour values
	xe = contour[0];   	ye = contour[1];   	nye = contour[2];
	xe2 = contour[3]; 	ye2 = contour[4];  	nye2 = contour[5];
	xed = contour[6];   yed = contour[7];   nyed = contour[8];
	xeca = contour[9]; yeca = contour[10]; nyeca = contour[11];
	xecc = contour[12]; yecc = contour[13]; nyecc = contour[14];
	if nozz_len == 0:
		xbell = contour[15]; ybell = contour[16]; nybell = contour[17];
	else:
		xec = contour[15]; yec = contour[16]; nyec = contour[17];
	
	# plot

	# set correct aspect ratio
	ax.set_aspect('equal')

	# throat enterant
	ax.plot(xe, ye, linewidth=2.5, color='g')
	ax.plot(xe, nye, linewidth=2.5, color='g')
	
	#convergent diagonal
	ax.plot(xed, yed, linewidth=2.5, color='k')
	ax.plot(xed, nyed, linewidth=2.5, color='k')

	#convergent arc
	ax.plot(xeca, yeca, linewidth=2.5, color='r')
	ax.plot(xeca, nyeca, linewidth=2.5, color='r')	

	#combustion chamber cylinder
	ax.plot(xecc, yecc, linewidth=2.5, color='b')
	ax.plot(xecc, nyecc, linewidth=2.5, color='b')	
	
	# throat inlet line
	x1 = xe[0]; y1 = 0;
	x2 = xe[0]; y2 = nye[0];
	dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
	# draw arrow, inlet radial line [x1, y1] to [x2, y2] 
	text = ' Ri = '+ str(round(dist,1))
	ax.plot(xe[0], 0, '+' )
	# draw dimension from [x1, y1] to [x2, y2] 
	ax.annotate( "", [x1, y1], [x2, y2] , arrowprops=dict(lw=0.5, arrowstyle='<-') )
	ax.text((x1+x2)/2, (y1+y2)/2, text, fontsize=9 )	

	# nozzle inlet length line [0,0] to [xe[0], 0]
	text = ' Li = ' + str( round( abs(xe[0]), 1) ) 
	ax.plot(0,0, '+' )
	# draw dimension from [0,0] to [xe[0], 0]
	ax.annotate( "", [0,0], [xe[0], 0], arrowprops=dict(lw=0.5, arrowstyle='<-') )
	ax.text( xe[0], 0, text, fontsize=9 )	
		
	# find mid point and draw arc radius
	i = int(len(xe)/2)
	xcenter = 0; 	ycenter = (1 + r1t) * Rt;  
	xarch = xe[i];  yarch = ye[i]
	# draw arrow, enterant radial line [xcenter, ycenter] to [xarch, yarch] 
	text =  str(r1t) +' * Rt = '+ str( round( r1t * Rt, 1 ) ) 
	ax.plot(xcenter, ycenter, '+' )
	# draw dimension from [xcenter, ycenter] to [xarch, yarch]
	ax.annotate( "", [xcenter, ycenter], [xarch, yarch], arrowprops=dict(lw=0.5, arrowstyle='<-') )
	ax.text((xarch+xcenter)/2, (yarch+ycenter)/2, text, fontsize=9 )	
		
	# throat radius line [0,0] to [xe[-1], ye[-1]]
	text = ' Rt = '+ str(Rt)
	# draw dimension from [0,0] to [xe[-1], ye[-1]]
	ax.annotate( "", [0,0], [xe[-1], ye[-1]], arrowprops=dict(lw=0.5, arrowstyle='<-') )
	ax.text( xe[-1]/2, ye[-1]/2, text, fontsize=9 )	

	# throat exit
	ax.plot(xe2, ye2, linewidth=2.5, color='r')
	ax.plot(xe2, nye2, linewidth=2.5, color='r')
	# find mid point and draw arc radius
	i = int(len(xe2)/2)
	xcenter2 = 0; 	ycenter2 = (1 + rnt) * Rt;  
	xarch2 = xe2[i];  yarch2 = ye2[i]
	# draw arrow, exit radial line from [xcenter2,ycenter2] to [xarch2, yarch2]
	text =  str(rnt) + ' * Rt = '+ str( round(rnt * Rt,1) ) 
	ax.plot(xcenter2, ycenter2, '+' )
	# draw dimension from [xcenter2,ycenter2] to [xarch2, yarch2]
	ax.annotate( "", [xcenter2,ycenter2], [xarch2, yarch2], arrowprops=dict(lw=0.5, arrowstyle='<-') )
	ax.text((xarch2+xcenter2)/2, (yarch2+ycenter2)/2, text, fontsize=9 )

	# draw theta_n, throat inflexion angle
	adj_text = 2
	origin	= [ xe2[-1], nye2[-1]-adj_text ]
	degree_symbol = r'$\theta$n'	
	draw_angle_arc(ax, np.rad2deg(theta_n), origin, degree_symbol )

	# bell section
	ax.plot(xbell, ybell, linewidth=2.5, color='b')
	ax.plot(xbell, nybell, linewidth=2.5, color='b')

	# throat radius line [0,0] to [xe[-1], ye[-1]]
	text = ' Re = ' + str( round( (math.sqrt(aratio) * Rt), 1) ) 
	ax.plot(xbell[-1],0, '+' )
	# draw dimension from [0,0] to [xe[-1], ye[-1]]
	ax.annotate( "", [xbell[-1],0], [xbell[-1], ybell[-1]], arrowprops=dict(lw=0.5, arrowstyle='<-') )
	ax.text( xbell[-1], ybell[-1]/2, text, fontsize=9 )	

	# draw theta_e, throat exit angle
	origin	= [ xbell[-1], nybell[-1] ]
	degree_symbol = r'$\theta$e'	
	draw_angle_arc(ax, (np.rad2deg(theta_e)), origin, degree_symbol )

	# nozzle length line [0,0] to [xe[-1], ye[-1]]
	text = ' Ln = ' + str( round( nozzle_length, 1) ) 
	ax.plot(0,0, '+' )
	# draw dimension from [0,0] to [xbell[-1], 0]
	ax.annotate( "", [0,0], [xbell[-1], 0], arrowprops=dict(lw=0.5, arrowstyle='<-') )
	ax.text( xbell[-1]/2, 0, text, fontsize=9 )	
				
	# axis
	ax.axhline(color='black', lw=0.5, linestyle="dashed")
	ax.axvline(color='black', lw=0.5, linestyle="dashed")		
	
	# grids
	ax.grid()
	ax.minorticks_on()
	ax.grid(which='major', linestyle='-', linewidth='0.5') # , color='red'
	ax.grid(which='minor', linestyle=':', linewidth='0.5') # , color='black'	
	
	# show
	plt.title(title, fontsize=9)
	return

# nozzle contour plot
def plot_nozzle_final(contour, angles, dia_t, dia_c, dia_e, len_c, Rad2, cangle, eangle, rt1, r2m, rnt, conic, unit):
	plt.close('all')
	# wall angles
	if conic == 0:
		theta_n = angles[1]; theta_e = angles[2];
	else:
		eangle = math.radians(eangle)

	if unit == "Inches":
		unit_s = "in"
		conv_o = 1 / in_to_m
	elif unit == "Centimeters":
		unit_s = "cm"
		conv_o = m_to_cm
	elif unit == "Millimeters":
		unit_s = "mm"
		conv_o = m_to_mm
	elif unit == "Meters":
		unit_s = "m"
		conv_o = 1

	bg_color = '#181618'

	R2 = Rad2 * conv_o

	xe    = np.array(contour[0])  * conv_o;  ye    = np.array(contour[1])  * conv_o;  nye   = np.array(contour[2])  * conv_o
	xe2   = np.array(contour[3])  * conv_o;  ye2   = np.array(contour[4])  * conv_o;  nye2  = np.array(contour[5])  * conv_o
	xed   = np.array(contour[6])  * conv_o;  yed   = np.array(contour[7])  * conv_o;  nyed  = np.array(contour[8])  * conv_o
	xeca  = np.array(contour[9])  * conv_o;  yeca  = np.array(contour[10]) * conv_o;  nyeca = np.array(contour[11]) * conv_o
	xecc  = np.array(contour[12]) * conv_o;  yecc  = np.array(contour[13]) * conv_o;  nyecc = np.array(contour[14]) * conv_o
	xnozz = np.array(contour[15]) * conv_o;  ynozz = np.array(contour[16]) * conv_o;  nynozz= np.array(contour[17]) * conv_o
	# plot

	fig2 = plt.figure(figsize=(12,9), facecolor='none')
	ax = fig2.add_subplot(111)
	ax.set_facecolor('none')
	ax.patch.set_alpha(0)          # ← makes axes patch fully transparent
	fig2.patch.set_alpha(0)        # ← makes figure patch fully transparent

	# Remove the axes border box
	for spine in ax.spines.values():
		spine.set_visible(False)   # ← hides the box around the plot area

	# throat enterant
	plt.plot(xe, ye, linewidth=2.5, color='#9100FF')
	plt.plot(xe, nye, linewidth=2.5, color='#9100FF')
	
	#convergent diagonal
	plt.plot(xed, yed, linewidth=2.5, color='#9100FF')
	plt.plot(xed, nyed, linewidth=2.5, color='#9100FF')

	#convergent arc
	plt.plot(xeca, yeca, linewidth=2.5, color='#9100FF')
	plt.plot(xeca, nyeca, linewidth=2.5, color='#9100FF')	

	#combustion chamber cylinder
	plt.plot(xecc, yecc, linewidth=2.5, color='#9100FF')
	plt.plot(xecc, nyecc, linewidth=2.5, color='#9100FF')	

	# throat exit
	plt.plot(xe2, ye2, linewidth=2.5, color='#9100FF')
	plt.plot(xe2, nye2, linewidth=2.5, color='#9100FF')

	# bell section
	plt.plot(xnozz, ynozz, linewidth=2.5, color='#9100FF')
	plt.plot(xnozz, nynozz, linewidth=2.5, color='#9100FF')

	# throat diameter line
	text = str(round(dia_t,2)) + unit_s
	# draw dimension from [0,0] to [xe[-1], ye[-1]]
	plt.annotate( "", [xe[-1], 0.95 * nye[-1]], [xe[-1], 0.95 * ye[-1]], arrowprops=dict(lw=1, arrowstyle='|-|', color='#ffffff'))
	plt.text(0.1,0.1, text, fontsize=25, color='#ffffff' )	

	# chamber diameter line
	text = str(round(dia_c,2)) + unit_s
	# draw dimension from [0,0] to [xe[-1], ye[-1]]
	plt.annotate( "", [xecc[-1], 0.95 * nyecc[-1]], [xecc[-1], 0.95 * yecc[-1]], arrowprops=dict(lw=1, arrowstyle='|-|', color='#ffffff') )
	plt.text(xecc[-1] + 0.1,0.1, text, fontsize=25, color='#ffffff' )	

	# exit diameter line
	text = str(round(dia_e,2)) + unit_s
	# draw dimension from [0,0] to [xe[-1], ye[-1]]
	plt.annotate( "", [xnozz[-1], 0.95 * nynozz[-1]], [xnozz[-1], 0.95 * ynozz[-1]], arrowprops=dict(lw=1, arrowstyle='|-|', color='#ffffff') )
	plt.text(4.0,0.1, text, fontsize=25, color='#ffffff' )	

	con_len = len_c - abs(xecc[0])

	# chamber length line
	text = str(round(con_len,2)) + unit_s
	# draw dimension from [0,0] to [xe[-1], ye[-1]]
	plt.annotate( "", [(xecc[-1] - 0.05), 1.2 * ynozz[-1]], [(xecc[0] + 0.05), 1.2 * ynozz[-1]], arrowprops=dict(lw=1, arrowstyle='|-|', color='#ffffff'))
	plt.text(-8.9, 1.25* ynozz[-1], text, fontsize=25, color='#ffffff')

	# convergent length line
	text = str(round(abs(xecc[0]),2)) + unit_s
	# draw dimension from [0,0] to [xe[-1], ye[-1]]
	plt.annotate( "", [(xecc[0] - 0.05), 1.2 * ynozz[-1]], [0.05, 1.2 * ynozz[-1]], arrowprops=dict(lw=1, arrowstyle='|-|', color='#ffffff'))
	plt.text(-2.0, 1.25* ynozz[-1], text, fontsize=25, color='#ffffff')

	# divergent section length line
	text = str(round(xnozz[-1],2)) + unit_s
	# draw dimension from [0,0] to [xe[-1], ye[-1]]
	plt.annotate( "", [-0.05, 1.2 * ynozz[-1]], [(xnozz[-1] + 0.05), 1.2 * ynozz[-1]], arrowprops=dict(lw=1, arrowstyle='|-|', color='#ffffff'))
	plt.text(2.0, 1.25* ynozz[-1], text, fontsize=25, color='#ffffff')

	if conic == 1:
		angle_list 		= np.linspace(0, eangle, 100)
		earcx = []; earcy = [];
		for i in angle_list:
			earcx.append( xe2[-1] + 2.0 * math.cos(i))
			earcy.append( ye2[-1] + 2.0 * math.sin(i))

		# theta exit angle
		text = r'$\theta_e$ = ' + str(round((eangle * 180 / np.pi),1)) + r'$^\circ$'
		# draw dimension from [0,0] to [xe[-1], ye[-1]]
		plt.annotate( "", [xe2[-1], ye2[-1]], [(xe2[-1] + 2.5), ye2[-1]], arrowprops=dict(lw=1, arrowstyle='-', color='#DBC885'))
		plt.annotate( "", [xe2[-1], ye2[-1]], [(xe2[-1] + (2.5 * math.cos(eangle))), (ye2[-1] + (2.5 * math.sin(eangle)))], arrowprops=dict(lw=1, arrowstyle='-', color='#DBC885'))
		plt.plot(earcx, earcy, linewidth=1, color='#DBC885')
		plt.text(2.5, 1.75, text, fontsize=20, color='#ffffff')
	else:
		# Theta n arc functions
		angle_list 		= np.linspace(0, theta_n, 100)
		tnarcx = []; tnarcy = [];
		for i in angle_list:
			tnarcx.append( xe2[-1] + 2.0 * math.cos(i))
			tnarcy.append( ye2[-1] + 2.0 * math.sin(i))

		# theta n angle
		text = r'$\theta_n$ = ' + str(round((theta_n * 180 / np.pi),1)) + r'$^\circ$'
		# draw dimension from [0,0] to [xe[-1], ye[-1]]
		plt.annotate( "", [xe2[-1], ye2[-1]], [(xe2[-1] + 2.5), ye2[-1]], arrowprops=dict(lw=1, arrowstyle='-', color='#DBC885'))
		plt.annotate( "", [xe2[-1], ye2[-1]], [(xe2[-1] + (2.5 * math.cos(theta_n))), (ye2[-1] + (2.5 * math.sin(theta_n)))], arrowprops=dict(lw=1, arrowstyle='-', color='#DBC885'))
		plt.plot(tnarcx, tnarcy, linewidth=1, color='#DBC885')
		plt.text(2.5, 1.75, text, fontsize=20, color='#ffffff')

		# Theta e arc functions
		angle_list 		= np.linspace(0, theta_e, 100)
		tearcx = []; tearcy = [];
		for i in angle_list:
			tearcx.append( xnozz[-1] + 1.5 * math.cos(i))
			tearcy.append( ynozz[-1] + 1.5 * math.sin(i))

		# theta e angle
		text = r'$\theta_e$ = ' + str(round((theta_e * 180 / np.pi),1)) + r'$^\circ$'
		# draw dimension from [0,0] to [xe[-1], ye[-1]]
		plt.annotate( "", [xnozz[-1], ynozz[-1]], [(xnozz[-1] + 2.0), ynozz[-1]], arrowprops=dict(lw=1, arrowstyle='-', color='#DBC885'))
		plt.annotate( "", [xnozz[-1], ynozz[-1]], [(xnozz[-1] + (2.0 * math.cos(theta_e))), (ynozz[-1] + (2.0 * math.sin(theta_e)))], arrowprops=dict(lw=1, arrowstyle='-', color='#DBC885'))
		plt.plot(tearcx, tearcy, linewidth=1, color='#DBC885')
		plt.text(6, 5, text, fontsize=20, color='#ffffff')

	#r1 distance line
	rt = dia_t / 2.0
	r1_hangle = ((270 + 180 + cangle) / 2) * np.pi / 180
	plt.annotate( "", [0, (2.5 * rt)], [(1.5 * rt * math.cos(r1_hangle)), (rt * ((1.5 * math.sin(r1_hangle)) + 2.5))], arrowprops=dict(lw=.5, arrowstyle='<-', color='#DBC885'))
	plt.plot(0, (2.5 * rt), '+' , color='#DBC885')
	plt.text(-1.0, 2.75, r'$R_1$', fontsize=15, color='#ffffff')

	#r2 distance line
	r2_hangle = ((90 + cangle) / 2) * np.pi / 180
	plt.annotate( "", [xecc[0], (yecc[0] - R2)], [(xecc[0] + R2 * math.cos(r2_hangle)), (yecc[0] + R2 * (math.sin(r2_hangle) - 1))], arrowprops=dict(lw=.5, arrowstyle='<-', color='#DBC885'))
	plt.plot(xecc[0], (yecc[0] - R2), '+' , color='#DBC885')
	plt.text(-3.1, 2.1, r'$R_2$', fontsize=15, color='#ffffff')	

	#rn distance line
	rn_hangle = ((270 + 270 + cangle) / 2) * np.pi / 180
	plt.annotate( "", [0, (1.382 * rt)], [(0.382 * rt * math.cos(rn_hangle)), (rt * ((0.382 * math.sin(rn_hangle)) + 1.382))], arrowprops=dict(lw=.5, arrowstyle='<-', color='#DBC885'))
	plt.plot(0, (1.382 * rt), '+', color='#DBC885')
	plt.text(0.1, 2.0, r'$R_n$', fontsize=15, color='#ffffff')

	# axis
	plt.axhline(color='#ffffff', lw=0.5, linestyle="dashed")
	plt.axvline(color='#ffffff', lw=0.5, linestyle="dashed")		
	
	# grids
	plt.grid(color='#ffffff')
	plt.minorticks_on()
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='#ffffff') # , color='red'
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='#ffffff') # , color='black'	
	plt.xticks(fontsize=15, color='#ffffff')
	plt.yticks(fontsize=15, color='#ffffff')
	
	# show
	plt.xlabel(unit, fontsize=18, color='#ffffff')
	plt.ylabel(unit, fontsize=18, color='#ffffff')
	plt.axis('equal')
	fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
	
	buf = io.BytesIO()
	plt.savefig(buf,
				format="png",
				transparent=True,
				facecolor='none',
				bbox_inches='tight')
	plt.close(fig2)
	buf.seek(0)
	return buf

# theta_n in rad,  origin =[startx, starty], degree symbol
def draw_angle_arc(ax, theta_n, origin, degree_symbol=r'$\theta$'):
	length = 50
	# start point
	startx = origin[0]; starty = origin[1];
	# find the end point
	endx = startx + np.cos(-theta_n) * length * 0.5
	endy = starty + np.sin(-theta_n) * length * 0.5
	# draw the angled line
	ax.plot([startx,endx], [starty,endy], linewidth=0.5, color='k')
	# horizontal line
	# ax.hlines(y=starty, xmin=startx, xmax=length, linewidth=0.5, color='k')
	# angle
	arc_obj = Arc([startx, starty], 1, 1, angle=0, theta1=0, theta2=math.degrees(theta_n), color='k' )
	ax.add_patch(arc_obj)
	ax.text(startx+0.5, starty+0.5, degree_symbol + ' = ' + str(round(theta_n,1)) + u"\u00b0")	
	return

# ring of radius r, height h, base point a
def ring(r, h, a=0, n_theta=30, n_height=10):
	theta = np.linspace(0, 2*np.pi, n_theta)
	v = np.linspace(a, a+h, n_height )
	theta, v = np.meshgrid(theta, v)
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	z = v
	return x, y, z

# Set 3D plot axes to equal scale. 
# Required since `ax.axis('equal')` and `ax.set_aspect('equal')` don't work on 3D.
def set_axes_equal_3d(ax: plt.Axes):
	"""	
	https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
	"""
	limits = np.array([
		ax.get_xlim3d(),
		ax.get_ylim3d(),
		ax.get_zlim3d(),
	])
	origin = np.mean(limits, axis=1)
	radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
	_set_axes_radius(ax, origin, radius)
	return

# set axis limits
def _set_axes_radius(ax, origin, radius):
	x, y, z = origin
	ax.set_xlim3d([x - radius, x + radius])
	ax.set_ylim3d([y - radius, y + radius])
	ax.set_zlim3d([z - radius, z + radius])
	return

# 3d plot
def plot3D_interactive(contour, unit):
	x = []; y = []
	
	if unit == "Inches":
		unit_s = "in"
		conv_o = 1 / in_to_m
	elif unit == "Centimeters":
		unit_s = "cm"
		conv_o = m_to_cm
	elif unit == "Millimeters":
		unit_s = "mm"
		conv_o = m_to_mm
	elif unit == "Meters":
		unit_s = "m"
		conv_o = 1

	xe    = np.array(contour[0])  * conv_o;  ye    = np.array(contour[1])  * conv_o
	xe2   = np.array(contour[3])  * conv_o;  ye2   = np.array(contour[4])  * conv_o
	xed   = np.array(contour[6])  * conv_o;  yed   = np.array(contour[7])  * conv_o
	xeca  = np.array(contour[9])  * conv_o;  yeca  = np.array(contour[10]) * conv_o
	xecc  = np.array(contour[12]) * conv_o;  yecc  = np.array(contour[13]) * conv_o
	xnozz = np.array(contour[15]) * conv_o;  ynozz = np.array(contour[16]) * conv_o

	for arr_x, arr_y in [(xecc,yecc),(xeca,yeca),(xed,yed),(xe,ye),(xe2,ye2),(xnozz,ynozz)]:
		x = np.append(x, arr_x)
		y = np.append(y, arr_y)

	# Build 3D surface by rotating the 2D contour around the axis
	n_theta = 60
	theta = np.linspace(0, 2 * np.pi, n_theta)

	X = np.outer(y, np.cos(theta))
	Y = np.outer(y, np.sin(theta))
	Z = np.outer(x, np.ones(n_theta))

	fig = go.Figure(data=[go.Surface(
		x=X, y=Y, z=Z,
		# Black → purple → white (matches your app theme)
		colorscale=[[0,'#9100FF'], [1, '#9100FF']],
		showscale=False,
		opacity=0.85
	)])

	fig.update_layout(
		scene=dict(
			xaxis_title=unit,
			yaxis_title=unit,
			zaxis_title=unit,
			aspectmode='data'
		),
		margin=dict(l=0, r=0, t=0, b=0),
		paper_bgcolor='black',
		scene_bgcolor='black' if False else None
	)

	return fig

######################### EXPORT TO CSV FILE #########################

def export_nozzle_csv(contour, con_factor):  # remove filename parameter
	if not isinstance(contour, (list, tuple)) or len(contour) < 9:
		raise ValueError("Unexpected contour structure. Need at least 9 elements as returned by bell_nozzle().")
	con = 1 / con_factor
	xe,   ye   = np.divide(contour[0], con), np.divide(contour[1], con)
	xe2,  ye2  = np.divide(contour[3], con), np.divide(contour[4], con)
	xed,  yed  = np.divide(contour[6], con), np.divide(contour[7], con)
	xeca, yeca = np.divide(contour[9], con), np.divide(contour[10], con)
	xecc, yecc = np.divide(contour[12], con), np.divide(contour[13], con)
	xbell,ybell= np.divide(contour[15], con), np.divide(contour[16], con)

	segments = [
		("chamber_wall", xecc, yecc),
		("convergent_arc", xeca, yeca),
		("convergent_diagonal", xed, yed),
		("throat_arc", xe,    ye),
		("inlet_arc",  xe2,   ye2),
		("bell",       xbell, ybell),
	]

	buf = io.StringIO()  # CSV is text, so StringIO not BytesIO
	w = csv.writer(buf)
	w.writerow(["segment","x","y","index"])
	for name, xs, ys in segments:
		n = min(len(xs), len(ys))
		for i in range(n):
			w.writerow([name, float(xs[i]), float(ys[i]), i])

	buf.seek(0)
	return buf

def export_nozzle_dxf(contour, con_factor):
	con = 1 / con_factor
	xe,   ye   = np.divide(contour[0], con), np.divide(contour[1], con)
	xe2,  ye2  = np.divide(contour[3], con), np.divide(contour[4], con)
	xed,  yed  = np.divide(contour[6], con), np.divide(contour[7], con)
	xeca, yeca = np.divide(contour[9], con), np.divide(contour[10], con)
	xecc, yecc = np.divide(contour[12], con), np.divide(contour[13], con)
	xbell,ybell= np.divide(contour[15], con), np.divide(contour[16], con)
	
	xed = xed[::-1];   yed = yed[::-1]
	xeca = xeca[::-1]; yeca = yeca[::-1]
	xecc = xecc[::-1]; yecc = yecc[::-1]

	doc = ezdxf.new("R2010")
	msp = doc.modelspace()

	sections = [
		(xecc, yecc),
		(xeca, yeca),
		(xed, yed),
		(xe, ye),
		(xe2, ye2),
		(xbell, ybell)
	]

	for x_vals, y_vals in sections:
		points = list(zip(x_vals, y_vals))
		if len(points) > 2:
			msp.add_spline(points, degree=3)

	# Write to StringIO first (DXF is text-based)
	str_buf = io.StringIO()
	doc.write(str_buf)
	
	# Convert to BytesIO for Streamlit
	bytes_buf = io.BytesIO(str_buf.getvalue().encode('utf-8'))
	bytes_buf.seek(0)
	return bytes_buf

##################################
#Runs all the code when the run button is hit
##################################

if run_contour:
	errors = []

	# Selectbox validation
	if ox == "-- select --":
		errors.append("⚠ Please select an Oxidizer")
	if fuel == "-- select --":
		errors.append("⚠ Please select a Fuel")
	if frozen == "-- select --":
		errors.append("⚠ Please select a Propellant State")
	if freeze_location == "-- select --":
		errors.append("⚠ Please select a Location of Freeze")

	# Number input validation
	if thrust_def == "Force of Thrust" and thrust is None:
		errors.append("⚠ Please enter a Thrust value")
	if thrust_def == "Force of Thrust" and thrust_unit is None:
		errors.append("⚠ Please enter a Thrust unit")
	if chamber_pressure is None:
		errors.append("⚠ Please enter a Chamber Pressure")
	if exit_pressure is None:
		errors.append("⚠ Please enter an Exit Pressure")
	if ambient_pressure is None:
		errors.append("⚠ Please enter an Ambient Pressure")
	if thrust_def == "Mass Flow Rate" and mass_flow is None:
		errors.append("⚠ Please enter a Mass Flow Rate")
	if thrust_def == "Mass Flow Rate" and mass_flow_unit is None:
		errors.append("⚠ Please enter a Mass Flow unit")
	if chamber_len_def == "Characteristic Length (L*)" and characteristic_length is None:
		errors.append("⚠ Please enter a Characteristic Length (L*)")
	if chamber_len_def == "Chamber Length (Lc)" and chamber_length is None:
		errors.append("⚠ Please enter a Chamber Length (Lc)")
	if characteristic_velo_eff is None:
		errors.append("⚠ Please enter a c* Efficiency")
	if coefficient_thrust_eff is None:
		errors.append("⚠ Please enter a Cf Efficiency")
	if chamber_size_def == "Chamber Diameter" and chamber_dia is None:
		errors.append("⚠ Please enter a Chamber Diameter")
	if chamber_size_def == "Contraction Ratio" and con_ratio is None:
		errors.append("⚠ Please enter a Contraction Ratio")
	if nozzle_type == "Bell Nozzle" and bell_percent is None:
		errors.append("⚠ Please enter a Bell Nozzle Length %")
	if nozzle_type == "Conic Nozzle" and divergent_angle is None:
		errors.append("⚠ Please enter a Divergent Angle")

	# Show errors or run
	if errors:
		for e in errors:
			st.error(e)
	else:
		with st.spinner("Running CEA and computing contour..."):
			try:
				#CEA Object Definition with user defined props
				if ox == "IPA":
					ox = "Isopropanol"
				
				C = CEA_Obj(oxName=ox, fuelName=fuel)

				of = of_ratio   #changes name of of_ratio to of

				#Adjusts mass flow rate to kg/s
				if thrust_def == "Mass Flow Rate":
					thrust = 0
					thrust_unit = 0
					if mass_flow_unit == "lb/s":
						mass_flow = mass_flow / kg_to_lbm
				else:
					mass_flow = 0
					mass_flow_unit = 0
					#Adjusts thrust to N
					if thrust_unit == "lbf":
						thrust = thrust * lbf_N

				#Debugging Chamber Length
				if chamber_len_def == "Chamber Length (Lc)":
					characteristic_length = 0
					characteristic_length_unit = 0
				else:
					chamber_length = 0
					chamber_length_unit = 0

				cstar_eff = characteristic_velo_eff / 100.0  #converts efficient factors to fractions instead of percentages, changes name
				cf_eff = coefficient_thrust_eff / 100.0  #converts efficient factors to fractions instead of percentages, changes name

				#Getting user input pressure into psi and pascals
				pc_psi, pc_pa = pressure_convert(chamber_pressure, chamber_pressure_unit)
				pe_psi, pe_pa = pressure_convert(exit_pressure, exit_pressure_unit)
				pamb_psi, pamb_pa = pressure_convert(ambient_pressure, ambient_pressure_unit)

				#Characterizing the frozen flags
				if frozen == "Equilibrium":
					f1 = 0
					f2 = 0
				elif frozen == "Frozen":
					f1 = 1
					if freeze_location == "Chamber":
						f2 = 0
					elif freeze_location == "Throat":
						f2 = 1

				if chamber_dia is None:
					chamber_dia = 0
				chamber_dia = conv_to_m(chamber_dia, chamber_dia_unit)
				chamber_length = conv_to_m(chamber_length, chamber_length_unit)
				characteristic_length = conv_to_m(characteristic_length, characteristic_length_unit)

				Ft_N, mdot_kgs, At_m, Eratio, Ae_m = te_sizing(thrust, mass_flow, of, pc_psi, pe_psi, pamb_psi, cstar_eff, cf_eff, f1, f2)
				Dc_m, conr, Lc_m = c_sizing(At_m, chamber_dia, con_ratio, characteristic_length, chamber_length, contraction_angle)

				isp = C.estimate_Ambient_Isp(Pc = pc_psi, MR = of, eps = Eratio, Pamb = pamb_psi, frozen=f1, frozenAtThroat=f2)

				# typical upper stage values
				aratio = Eratio	
				cratio = conr
				cangle = contraction_angle
				clength = Lc_m    #* m_to_mm
				throat_radius = np.sqrt(At_m / np.pi)     #* m_to_mm		#throat radius (mm)

				###CHANGE TO EXPORT WHOLE CONTOUR
				if nozzle_type == "Conic Nozzle":
					contour, r2, Ln = conic_nozzle(aratio, divergent_angle, throat_radius, cratio, cangle, clength, radius1, radius2, radiusn)
					#(aratio, eangle, Rt, cratio, alpha, Lc, r1t, r2m, rnt)
					ntype = 1
					angles = [0, 0, 0]
				else:
					l_percent = bell_percent
					# rao_bell_nozzle_contour
					angles, contour, r2, Ln = bell_nozzle(aratio, throat_radius, l_percent, cratio, cangle, clength, radius1, radius2, radiusn)
					ntype = 0
					##################################
				#REPLACE PATH WITH PATH YOU NEED FOR YOUR OWN COMPUTER
				#Devin Path: "C:\Users\igoto\Downloads\GH\Turbopump\TCA\Countour Exports\nozzle_contour.csv"
				#Dani Path: "/Users/dl/Documents/GitHub/Turbopump/TCA/Countour Exports/nozzle_contour.csv"
				#Other Path: 
				#################################

				#Need Dt, Dc, De, Lc in inches
				if out_len_unit == "in":
					conv_o = 1 / in_to_m
					graph_unit = "Inches"
				elif out_len_unit == "mm":
					conv_o = m_to_mm
					graph_unit = "Millimeters"
				elif out_len_unit == 'cm':
					conv_o = m_to_cm
					graph_unit = "Centimeters"
				elif out_len_unit == 'm':
					conv_o = 1
					graph_unit = "Meters"

				#Get nozzle design functions to b e implemented
				if csv_plot == True:
					csv_gen = export_nozzle_csv(contour, conv_o)

				if dxf_plots == True:
					dxf_gen = export_nozzle_dxf(contour, conv_o)
				
				Dt_o = 2 * np.sqrt(At_m / np.pi) * conv_o
				Dc_o = Dc_m * conv_o
				De_o = 2 * np.sqrt(Ae_m / np.pi) * conv_o
				con_ratio = (Dc_o / Dt_o) * 2
				Lc_o = Lc_m * conv_o
				L_nozz = Ln * conv_o
				Lstar_o = characteristic_length  * conv_o
				thetan = angles[1] * 180 / np.pi
				thetae = angles[2] * 180 /np.pi
				mdot = 0
				Ft = 0
				if out_m_unit == "lb":
					mdot = mdot_kgs * kg_to_lbm
					Ft = Ft_N / lbf_N
					mdot_unit = "lb/s"
					force_unit = "lbf"
				elif out_m_unit == "kg":
					mdot = mdot_kgs
					Ft = Ft_N
					mdot_unit = "kg/s"
					force_unit = "N"

				twodplot = plot_nozzle_final(contour, angles, Dt_o, Dc_o, De_o, Lc_o, r2, cangle, divergent_angle, radius1, radius2, radiusn, ntype, graph_unit)
				#(contour, angles, dia_t, dia_c, dia_e, len_c, Rad2, cangle, eangle, rt1, r2m, rnt, conic)

				st.success("✓ Nozzle Generation Complete!")
				st.session_state.computed = True
				
				#SAVING SESSION STATES
				st.session_state.ox = ox
				st.session_state.of = of
				st.session_state.pc_psi = pc_psi
				st.session_state.pc_pa = pc_pa
				st.session_state.pe_psi = pe_psi
				st.session_state.pe_pa = pe_pa
				st.session_state.pamb_psi = pamb_psi
				st.session_state.pamb_pa = pamb_pa
				st.session_state.f1 = f1
				st.session_state.f2 = f2
				st.session_state.cstar_eff = cstar_eff
				st.session_state.cf_eff = cf_eff
				st.session_state.Ft_N = Ft_N
				st.session_state.Ft = Ft
				st.session_state.mdot_kgs = mdot_kgs
				st.session_state.mdot = mdot
				st.session_state.mdot_unit = mdot_unit
				st.session_state.force_unit = force_unit
				st.session_state.At_m = At_m
				st.session_state.Ae_m = Ae_m
				st.session_state.Eratio = Eratio
				st.session_state.Dc_m = Dc_m
				st.session_state.Lc_m = Lc_m
				st.session_state.conv_o = conv_o
				st.session_state.Dt_o = Dt_o
				st.session_state.Dc_o = Dc_o
				st.session_state.De_o = De_o
				st.session_state.Lc_o = Lc_o
				st.session_state.Lstar_o = Lstar_o
				st.session_state.L_nozz = L_nozz
				st.session_state.contour = contour
				st.session_state.isp = isp
				st.session_state.thetan = thetan
				st.session_state.thetae = thetae
				st.session_state.con_ratio = con_ratio
				st.session_state.graph_unit = graph_unit
				st.session_state.twodplot = twodplot
				st.session_state.conr = conr

				if twod_contour == True:
					st.session_state.twodimage = twodplot
				if threed_contour == True:
					threed_fig = plot3D_interactive(contour, graph_unit)
					st.session_state.threedimage = threed_fig
					st.session_state.threed_is_plotly = True
				if csv_plot == True:
					st.session_state.csvfile = csv_gen
				if dxf_plots == True:
					st.session_state.dxffile = dxf_gen
			except Exception as e:
				import traceback
				st.error(f"Computation failed: {e}")
				#st.code(traceback.format_exc())

################################
#Nozzle Contour Outputs Tab
################################

with tab2:
	if st.session_state.get("computed"):
		ox = st.session_state.ox
		of = st.session_state.of
		pc_psi = st.session_state.pc_psi
		pc_pa = st.session_state.pc_pa
		pe_psi = st.session_state.pe_psi
		pe_pa = st.session_state.pe_pa
		pamb_psi = st.session_state.pamb_psi
		pamb_pa = st.session_state.pamb_pa
		f1 = st.session_state.f1
		f2 = st.session_state.f2
		cstar_eff = st.session_state.cstar_eff
		cf_eff = st.session_state.cf_eff
		Ft_N = st.session_state.Ft_N
		Ft = st.session_state.Ft
		mdot_kgs = st.session_state.mdot_kgs
		mdot = st.session_state.mdot
		mdot_unit = st.session_state.mdot_unit
		force_unit = st.session_state.force_unit
		At_m = st.session_state.At_m
		Ae_m = st.session_state.Ae_m
		Eratio = st.session_state.Eratio
		Dc_m = st.session_state.Dc_m
		Lc_m = st.session_state.Lc_m
		conv_o = st.session_state.conv_o
		Dt_o = st.session_state.Dt_o
		Dc_o = st.session_state.Dc_o
		De_o = st.session_state.De_o
		Lc_o = st.session_state.Lc_o
		Lstar_o = st.session_state.Lstar_o
		L_nozz = st.session_state.L_nozz
		contour = st.session_state.contour
		isp = st.session_state.isp
		thetan = st.session_state.thetan
		thetae = st.session_state.thetae
		con_ratio = st.session_state.con_ratio
		graph_unit = st.session_state.graph_unit
		twodplot = st.session_state.twodplot
		conr = st.session_state.conr

		if "twodimage" in st.session_state and "threedimage" in st.session_state:
			col1, col2 = st.columns([7, 4])
			with col1:
				st.session_state.twodimage.seek(0)
				st.image(st.session_state.twodimage, use_container_width=True)
			with col2:
				st.plotly_chart(st.session_state.threedimage, use_container_width=True)
		elif "twodimage" in st.session_state:
			st.session_state.twodimage.seek(0)
			st.image(st.session_state.twodimage, use_container_width=True)
		elif "threedimage" in st.session_state:
			st.plotly_chart(st.session_state.threedimage, use_container_width=True)
		else:
			st.info("No plots generated — check your output options in the sidebar.")

		col1, col2, col3, col4 = st.columns(4)

		with col3:
			if st.session_state.get("csvfile"):
				st.session_state.csvfile.seek(0)
				st.download_button(
					label="⬇ Download CSV",
					data=st.session_state.csvfile.getvalue().encode('utf-8'),  # ← encode to bytes
					file_name="nozzle_contour.csv",
					mime="text/csv",
					key="dl_csv"
				)
			else:
				st.button("⬇ Download CSV", disabled=True, key="dl_csv")

		with col4:
			if st.session_state.get("dxffile"):
				st.session_state.dxffile.seek(0)
				st.download_button(
					label="⬇ Download DXF",
					data=st.session_state.dxffile,
					file_name="nozzle_contour.dxf",
					mime="application/dxf",
					key="dl_dxf"
				)
			else:
				st.button("⬇ Download DXF", disabled=True, key="dl_dxf")

		with col1:
			if st.session_state.get("twodimage"):
				st.session_state.twodimage.seek(0)
				st.download_button(
					label="⬇ Download 2D Plot",
					data=st.session_state.twodimage,
					file_name="nozzle_contour_2d.png",
					mime="image/png",
					key="dl_2d"
				)
			else:
				st.button("⬇ Download 2D Plot", disabled=True, key="dl_2d")
		with col2:
			if st.session_state.get("threedimage"):
				html_bytes = st.session_state.threedimage.to_html(include_plotlyjs='cdn').encode('utf-8')
				st.download_button(
					label="⬇ Download 3D Plot",
					data=html_bytes,
					file_name="nozzle_contour_3d.html",
					mime="text/html",
					key="dl_3d"
					)
			else:
				st.button("⬇ Download 3D Plot", disabled=True, key="dl_3d")

		st.divider()

		st.markdown("""
			<style>
				/* Vertical dividers — white */
				[data-testid="stColumn"] {
					border-right: 1px solid #ffffff;
					padding: 0 10px;
				}
				[data-testid="stColumn"]:last-child {
					border-right: none;
				}
				/* Metric value — purple */
				[data-testid="stMetricValue"] {
					color: #9100FF !important;
				}
				/* Metric label — white */
				[data-testid="stMetricLabel"] {
					color: #ffffff !important;
				}
			</style>
		""", unsafe_allow_html=True)

		st.divider()

		if thrust_def == "Force of Thrust":
			t_mf_def1 = "Force of Thrust"
			t_mf1 = Ft
			t_mf_unit1 = force_unit
			t_mf_def2 = "Mass Flow Rate"
			t_mf2 = mdot
			t_mf_unit2 = mdot_unit
		elif thrust_def == "Mass Flow Rate":
			t_mf_def2 = "Force of Thrust"
			t_mf2 = Ft
			t_mf_unit2 = force_unit
			t_mf_def1 = "Mass Flow Rate"
			t_mf1 = mdot
			t_mf_unit1 = mdot_unit
		
		if frozen == "Equilibrium":
			p_state = "Equilibrium"
		else:
			if freeze_location == "Chamber":
				p_state = "Frozen in Chamber"
			else:
				p_state = "Frozen at Throat"

		if out_p_unit == "psia":
			pc = pc_pa / psi_to_pa
			pe = pe_pa / psi_to_pa
			pamb = pamb_pa / psi_to_pa
		elif out_p_unit == "Pa":
			pc = pc_pa
			pe = pe_pa
			pamb = pamb_pa
		elif out_p_unit == "kPa":
			pc = pc_pa / (psi_to_pa * kpa_to_pa)
			pe = pe_pa / (psi_to_pa * kpa_to_pa)
			pamb = pamb_pa / (psi_to_pa * kpa_to_pa)
		elif out_p_unit == "atm":
			pc = pc_pa / (atm_to_pa)
			pe = pe_pa / (atm_to_pa)
			pamb = pamb_pa / (atm_to_pa)
		elif out_p_unit == "bar":
			pc = pc_pa / (bar_to_pa)
			pe = pe_pa / (bar_to_pa)
			pamb = pamb_pa / (bar_to_pa)

		if chamber_size_def == "Chamber Diameter":
			cdef_id1 = "Chamber Diameter"
			cdef_val1 = Dc_o
			cdef_unit1 = out_len_unit
			cdef_id2 = "Contraction Ratio"
			cdef_val2 = conr
			cdef_unit2 = "—"
		else:
			cdef_val1 = conr
			cdef_unit1 = "—"
			cdef_id2 = "Chamber Diameter"
			cdef_val2 = Dc_o
			cdef_unit2 = out_len_unit
			cdef_id1 = "Contraction Ratio"

		if chamber_len_def == "Characteristic Length (L*)":
			cl_id1 = "Characteristic Length (L*)"
			cl_val1 = Lstar_o
			cl_id2 = "Chamber Length (Lc)"
			cl_val2 = Lc_o
		else:
			cl_id2 = "Characteristic Length (L*)"
			cl_val2 = Lstar_o
			cl_id1 = "Chamber Length (Lc)"
			cl_val1 = Lc_o

		if nozzle_type == "Bell Nozzle":
			nz = "Bell Percentage Length"
			nz_val = bell_percent
			nz_unit = "%"
			tn_def = "θn"
			te_def = "θe"
			tn = thetan
			te = thetae
			tn_u = "°"
			te_u = "°"
		else:
			nz = "Divergent Angle"
			nz_val = divergent_angle
			nz_unit = "°"
			tn_def = "—"
			te_def = "—"
			tn = "—"
			te = "—"
			tn_u = "—"
			te_u = "—"
		
		# Key metrics at top
		col1, col2, col3, col4 = st.columns(4)
		col1.metric("Throat Diameter",    f"{Dt_o:.3f}" + out_len_unit)
		col2.metric("Exit Diameter",      f"{De_o:.3f}" +out_len_unit)
		col3.metric("Estimated Ambient Isp",   f"{isp[0]:.1f}s")
		col4.metric("Generated Thrust",    f"{Ft:.0f}" + force_unit)

		# Full results table below
		col1, col2 = st.columns(2)

		with col1:
			st.subheader("All Input Parameters")
			df1 = pd.DataFrame({
				"Input Parameter": [
					"Fuel", "Oxidizer", "O/F Ratio", "Nozzle Type", "Propellant State", t_mf_def1, "c* Efficiency", "cf efficiency",
					"Chamber Pressure", "Exit Pressure", "Ambient Pressure", cdef_id1, "Contraction Angle",
					cl_id1, nz, "R1/Rt", "R2/R2max", "Rn/Rt"
				],
				"Value": [
					fuel, ox, round(of_ratio,2), nozzle_type, p_state, round(t_mf1,2), round(characteristic_velo_eff,2), 
					round(characteristic_velo_eff,2), round(pc,3), round(pe,3), round(pamb,3), round(cdef_val1,3), 
					round(contraction_angle,3), round(cl_val1,3), round(nz_val, 3), radius1, radius2, radiusn
				],
				"Units": [
					"—", "—", "—", "—", "—", t_mf_unit1, "%", "%", out_p_unit, out_p_unit, out_p_unit, cdef_unit1, "°", out_len_unit,
					nz_unit, "—", "—", "—"
				]
			})
			st.dataframe(
				df1.style
					.map(lambda v: 'font-weight: bold; color: #9100FF', subset=['Input Parameter'])
					.map(lambda v: 'color: #ffffff', subset=['Value', 'Units'])
					.format({'Value': lambda v: str(v) if not isinstance(v, float) else f'{v:g}'}),
				use_container_width=True,
				hide_index=True
			)
			st.caption("scroll for more...")
		with col2:
			st.subheader("All Output Parameters")
			df2 = pd.DataFrame({
				"Output Parameter": [
					"Throat Diameter", "Exit Diameter", cdef_id2, "Expansion Ratio", "Nozzle Length", "Total Length", cl_id2, t_mf_def2,
					tn_def, te_def
				],
				"Value": [
					round(Dt_o,3), round(De_o,3), round(cdef_val2,3), round(Eratio, 3), round(L_nozz,3), round((Lc_o + L_nozz),3), round(cl_val2,3),
					round(t_mf2,1), tn, te
				],
				"Units": [
					out_len_unit, out_len_unit, cdef_unit2, "—", out_len_unit, out_len_unit, out_len_unit, t_mf_unit2, tn_u, te_u
				]
			})
			st.dataframe(
				df2.style
					.map(lambda v: 'font-weight: bold; color: #9100FF', subset=['Output Parameter'])
					.map(lambda v: 'font-weight: bold; color: #ffffff', subset=['Value', 'Units'])
					.format({'Value': lambda v: str(v) if not isinstance(v, float) else f'{v:g}'}),
				use_container_width=True,
				hide_index=True
			)
			st.caption("scroll for more...")
	else:
		st.info("Run the solver to generate results.")

################################
#Nozzle Contour Outputs Tab
################################

with tab3:
	st.divider()
	if st.session_state.get("computed"):
		ox = st.session_state.ox
		of = st.session_state.of
		pc_psi = st.session_state.pc_psi
		pc_pa = st.session_state.pc_pa
		pe_psi = st.session_state.pe_psi
		pe_pa = st.session_state.pe_pa
		pamb_psi = st.session_state.pamb_psi
		pamb_pa = st.session_state.pamb_pa
		f1 = st.session_state.f1
		f2 = st.session_state.f2
		cstar_eff = st.session_state.cstar_eff
		cf_eff = st.session_state.cf_eff
		Ft_N = st.session_state.Ft_N
		Ft = st.session_state.Ft
		mdot_kgs = st.session_state.mdot_kgs
		mdot = st.session_state.mdot
		mdot_unit = st.session_state.mdot_unit
		force_unit = st.session_state.force_unit
		At_m = st.session_state.At_m
		Ae_m = st.session_state.Ae_m
		Eratio = st.session_state.Eratio
		Dc_m = st.session_state.Dc_m
		Lc_m = st.session_state.Lc_m
		conv_o = st.session_state.conv_o
		Dt_o = st.session_state.Dt_o
		Dc_o = st.session_state.Dc_o
		De_o = st.session_state.De_o
		Lc_o = st.session_state.Lc_o
		Lstar_o = st.session_state.Lstar_o
		L_nozz = st.session_state.L_nozz
		contour = st.session_state.contour
		isp = st.session_state.isp
		thetan = st.session_state.thetan
		thetae = st.session_state.thetae
		con_ratio = st.session_state.con_ratio
		graph_unit = st.session_state.graph_unit
		twodplot = st.session_state.twodplot
		conr = st.session_state.conr

		st.text("Below are your propellants' combustion characteristics throughout your nozzle.")
		# --- Unpack inputs ---
		all_params = [of, pc_psi, pe_psi, con_ratio, cp_conversion, tc_conversion, 
				visc_conversion, ox, fuel, f1, f2, Eratio, out_p_unit]
		df_properties, df_therm, df_poly = Bartz_Values.run(all_params)
		df_therm_display = df_therm.set_index('Station').T.reset_index()
		df_therm_display = df_therm_display.rename(columns={'index': 'Property'})
		st.dataframe(
			df_therm_display.style
				.map(lambda v: 'font-weight: bold; color: #9100FF', subset=['Property'])
				.map(lambda v: 'font-weight: bold; color: #ffffff', subset=['Injector', 'Combustor', 'Throat', 'Exit'])
				.format(subset=['Injector', 'Combustor', 'Throat', 'Exit'],
					formatter=lambda v: f'{v:.5g}' if isinstance(v, float) else str(v)),
			use_container_width=True,
			hide_index=True
		)
	else:
		st.info("Run the solver to generate results.")


