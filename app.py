import streamlit as st
import geemap.foliumap as geemap
import ee
import os
from samgeo import SamGeo2, tms_to_geotiff
import geopandas as gpd
import tempfile
import rasterio
from rasterio.plot import show
import numpy as np
import pandas as pd # Added missing import for pd.to_datetime
from google.oauth2 import service_account

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Crop Boundary Detection (SAM 2/3)")

st.title("🌱 Crop Field Delineation with GEE & SAM")
st.markdown("""
This app extracts crop boundaries from satellite imagery using **Google Earth Engine** and **Meta's Segment Anything Model (SAM 2)**. 
*Note: SAM 3 checkpoints can be used here once supported by the underlying libraries.*
""")

# --- Sidebar & Controls ---
with st.sidebar:
    st.header("1. Configuration")
    
    # GEE Initialization with Service Account
    try:
        if "gcp_service_account" in st.secrets:
            # 1. Load secrets
            service_account_info = dict(st.secrets["gcp_service_account"])
            
            # 2. CRITICAL FIX: Handle newline characters in private key
            # TOML/Streamlit sometimes escapes \n as literal characters. We fix this here.
            if "private_key" in service_account_info:
                service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

            # 3. Create Credentials
            creds = service_account.Credentials.from_service_account_info(service_account_info)
            
            # 4. Initialize Earth Engine
            # We explicitly pass the project_id to ensure correct billing/usage attribution
            project_id = service_account_info.get("project_id")
            ee.Initialize(credentials=creds, project=project_id)
            
            st.success(f"GEE Initialized! \nProject: {project_id}")
        else:
            st.warning("Secrets [gcp_service_account] not found. Trying default auth...")
            ee.Initialize()
            st.success("GEE Initialized (Default)")
            
    except Exception as e:
        st.error(f"GEE Initialization failed: {e}")
        st.info("Check your .streamlit/secrets.toml file. Ensure the 'private_key' is correct.")
        st.stop()

    st.divider()
    
    st.header("2. Data Parameters")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-06-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-06-30"))
    cloud_cover = st.slider("Max Cloud Cover (%)", 0, 30, 10)
    
    st.divider()
    
    st.header("3. Model Settings")
    model_type = st.selectbox(
        "SAM Model Type", 
        ["sam2_hiera_large", "sam2_hiera_small", "sam2_hiera_tiny"],
        index=1,
        help="Larger models are more accurate but require more GPU VRAM."
    )
    
    # Explanation for SAM 3
    st.info("For SAM 3: Once 'segment-geospatial' updates, select the SAM 3 checkpoint here.")

# --- Main Logic ---

# 1. Upload KML
st.subheader("1. Upload Area of Interest (KML)")
uploaded_kml = st.file_uploader("Upload a KML file defining your AOI", type=['kml'])

if uploaded_kml:
    # Save KML to temp file to read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp_kml:
        tmp_kml.write(uploaded_kml.getvalue())
        kml_path = tmp_kml.name

    # Convert KML to EE Geometry
    try:
        # We use geemap to convert KML to EE object
        gpd_df = gpd.read_file(kml_path)
        # Reproject to WGS84 if needed
        if gpd_df.crs != "EPSG:4326":
            gpd_df = gpd_df.to_crs("EPSG:4326")
        
        # Convert the first polygon to GEE geometry
        # (Assuming simple KML with one main polygon for the field)
        aoi_coords = list(gpd_df.geometry[0].exterior.coords)
        aoi = ee.Geometry.Polygon(aoi_coords)
        
        st.success(f"AOI Loaded. Centroid: {gpd_df.geometry[0].centroid}")
    except Exception as e:
        st.error(f"Error parsing KML: {e}")
        st.stop()

    # 2. Fetch Sentinel-2 Imagery
    st.subheader("2. Satellite Imagery (Sentinel-2)")
    
    def get_sentinel_image(roi, start, end, clouds):
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate(str(start), str(end)) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', clouds)) \
            .median() \
            .clip(roi)
        return s2

    s2_image = get_sentinel_image(aoi, start_date, end_date, cloud_cover)
    
    # Visual Parameters
    vis_params = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2'] # RGB
    }
    
    # Preview Map
    m = geemap.Map()
    m.centerObject(aoi, 14)
    m.addLayer(s2_image, vis_params, 'Sentinel-2 Image')
    m.addLayer(aoi, {'color': 'red'}, 'AOI')
    
    # Display map
    st.markdown("#### AOI & Satellite Preview")
    m.to_streamlit(height=500)

    # 3. Run SAM Segmentation
    st.subheader("3. Run Segmentation")
    
    if st.button("Run SAM to Detect Boundaries"):
        with st.spinner("Downloading image tile and initializing SAM model... (This may take a moment)"):
            try:
                # A. Download Image locally for SAM
                # SAM works on local numpy arrays, so we export the GEE image to a geotiff
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, 'satellite_image.tif')
                
                # Download the RGB bands as a GeoTIFF
                geemap.ee_export_image(
                    s2_image.select(['B4', 'B3', 'B2']),
                    filename=image_path,
                    scale=10, # 10m resolution for Sentinel-2
                    region=aoi,
                    file_per_band=False
                )
                
                # B. Initialize SAM
                # SAM 2 Initialization
                sam = SamGeo2(
                    model_id=model_type,
                    automatic=True, # Automatic Mask Generator
                    device='cuda' if st.checkbox("Use GPU (if available)", value=False) else 'cpu'
                )
                
                # C. Generate Masks
                output_mask_path = os.path.join(temp_dir, 'segmentation_mask.tif')
                output_vector_path = os.path.join(temp_dir, 'field_boundaries.gpkg')
                
                st.text("Running inference...")
                # Generate masks
                sam.generate(
                    source=image_path,
                    output=output_mask_path,
                    foreground=True,
                    unique=True
                )
                
                # Save as Vector (GeoPackage)
                sam.tiff_to_vector(output_mask_path, output_vector_path)
                
                # D. Display Results
                st.success("Segmentation Complete!")
                
                # Load the result vector to display
                gdf_result = gpd.read_file(output_vector_path)
                
                # Create result map
                m_result = geemap.Map()
                m_result.centerObject(aoi, 14)
                m_result.addLayer(s2_image, vis_params, 'Sentinel-2 Source')
                m_result.add_gdf(gdf_result, layer_name="Detected Fields", style_color="yellow", fill_colors=["rgba(0,0,0,0)"])
                
                st.markdown("#### Detected Boundaries")
                m_result.to_streamlit(height=500)
                
                # E. Download Options
                with open(output_vector_path, "rb") as f:
                     st.download_button(
                         label="Download Boundaries (GeoPackage)",
                         data=f,
                         file_name="field_boundaries.gpkg",
                         mime="application/octet-stream"
                     )
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.info("Tip: If running on CPU, try a smaller AOI or the 'Tiny' model.")

else:
    st.info("Please upload a KML file to begin.")
