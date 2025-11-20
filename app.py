import streamlit as st
import ee
import geemap.foliumap as geemap
import os
import tempfile

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AgriBoundary: Field Detector")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: 700;
    }
    .info-box {
        background-color: #e1f5fe;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0288d1;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Title & Intro ---
st.markdown('<div class="main-header">🌾 AgriBoundary Detector</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <b>How it works:</b><br>
    1. Upload a KML file containing your Area of Interest (AOI).<br>
    2. The app fetches Sentinel-2 satellite imagery for that area.<br>
    3. It applies the SNIC segmentation algorithm to detect field boundaries.<br>
    4. You can download the resulting boundaries as a KML file.
</div>
""", unsafe_allow_html=True)

# --- GEE Initialization ---
def initialize_gee():
    try:
        ee.Initialize()
        return True
    except Exception as e:
        st.warning("GEE not initialized. Trying to authenticate...")
        try:
            ee.Authenticate()
            ee.Initialize()
            return True
        except Exception as e2:
            st.error(f"Authentication failed: {e2}")
            st.info("Please run `earthengine authenticate` in your terminal first.")
            return False

if not initialize_gee():
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Parameters")
    
    # Date Range
    start_date = st.date_input("Start Date", value=import_datetime_date(2023, 5, 1))
    end_date = st.date_input("End Date", value=import_datetime_date(2023, 9, 30))
    
    # Cloud Filter
    cloud_pct = st.slider("Max Cloud Cover %", 0, 30, 10)
    
    # Segmentation Parameters (SNIC)
    st.subheader("Segmentation Tuning")
    seed_grid_size = st.slider("Grid/Seed Size (Pixels)", 10, 100, 30, help="Smaller = smaller fields, Larger = larger fields")
    compactness = st.slider("Compactness", 0.0, 2.0, 0.5, help="Shape vs Color importance")
    
    st.divider()
    st.caption("Powered by Google Earth Engine & Streamlit")

# --- Helper Functions ---

def get_sentinel_image(geometry, start, end, cloud_max):
    """Fetches and masks Sentinel-2 imagery."""
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
            qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000)

    dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(str(start), str(end)) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_max)) \
        .filterBounds(geometry) \
        .map(mask_s2_clouds)
    
    # Return the median composite to minimize clouds/artifacts
    return dataset.median().clip(geometry)

def detect_boundaries(image, geometry, size, compact):
    """Applies SNIC segmentation to detect boundaries."""
    
    # Select bands for segmentation (Visible + NIR usually best for fields)
    bands = ['B2', 'B3', 'B4', 'B8']
    input_image = image.select(bands)
    
    # Create seeds
    seeds = ee.Algorithms.Image.Segmentation.seedGrid(size)
    
    # Run SNIC
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=input_image, 
        compactness=compact,
        connectivity=8,
        neighborhoodSize=2 * size,
        seeds=seeds
    )
    
    clusters = snic.select('clusters')
    
    # Vectorize the clusters (Raster -> Vector)
    vectors = clusters.reduceToVectors(
        geometry=geometry,
        scale=10, # Sentinel-2 resolution
        geometryType='polygon',
        eightConnected=False,
        labelProperty='cluster_id'
    )
    
    return vectors, snic.select(bands).reproject(crs=snic.projection(), scale=10)

# --- Main Logic ---

uploaded_file = st.file_uploader("Upload AOI (KML file)", type=['kml'])

# Map initialization
m = geemap.Map(height=600)
m.add_basemap("HYBRID")

if uploaded_file is not None:
    try:
        # Save uploaded KML to a temp file so geemap can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.kml') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Convert KML to EE Geometry
        # Note: geemap uses internal logic to parse KML. If complex, might need fiona.
        # We assume simple polygon KML here.
        aoi_ee = geemap.kml_to_ee(tmp_path)
        
        # If aoi_ee is a FeatureCollection, get the geometry union
        if isinstance(aoi_ee, ee.FeatureCollection):
            aoi_geometry = aoi_ee.geometry()
        else:
            aoi_geometry = aoi_ee

        # Center Map
        m.centerObject(aoi_geometry, 13)
        
        with st.spinner('Processing Satellite Imagery...'):
            # 1. Get Imagery
            s2_image = get_sentinel_image(aoi_geometry, start_date, end_date, cloud_pct)
            
            # Display True Color Image
            vis_params = {'min': 0.0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
            m.addLayer(s2_image, vis_params, 'Sentinel-2 Imagery')
            
            # 2. Run Segmentation
            vectors, snic_raster = detect_boundaries(s2_image, aoi_geometry, seed_grid_size, compactness)
            
            # Display Segmentation
            m.addLayer(vectors, {'color': 'red', 'width': 2}, 'Detected Boundaries')

        # Show Map
        m.to_streamlit(width=None)
        
        # Results Section
        st.success("Processing Complete!")
        
        # Download Section
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate KML Download Link from GEE
            try:
                download_url = vectors.getDownloadURL(
                    filetype='kml', 
                    filename='detected_boundaries'
                )
                st.markdown(f"[**Click here to download Boundaries (KML)**]({download_url})")
                st.info("Note: This link is generated by Google Earth Engine servers.")
            except Exception as e:
                st.error(f"Error generating download link: {e}")

        # Cleanup temp file
        os.unlink(tmp_path)

    except Exception as e:
        st.error(f"An error occurred processing the file: {e}")
        st.warning("Ensure your KML contains a valid Polygon geometry.")
else:
    # Default map view
    m.to_streamlit()
    st.info("Please upload a KML file to start.")

def import_datetime_date(y, m, d):
    from datetime import date
    return date(y, m, d)
