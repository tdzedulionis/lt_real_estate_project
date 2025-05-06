import streamlit as st
import pandas as pd

def basic_info_inputs():
    """Create input fields for basic property information."""
    st.subheader("Basic Property Information")
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.selectbox(
            "City",
            options = [
            "Akmenės m.",
            "Akmenės r. sav.",
            "Alytaus r. sav.",
            "Alytaus k.",
            "Alytus",
            "Anykščių r. sav.",
            "Anykščių m.",
            "Birštono m.",
            "Biržų m.",
            "Biržų k.",
            "Biržų r. sav.",
            "Druskininkų m.",
            "Druskininkų sav.",
            "Elektrėnų m.",
            "Elektrėnų sav.",
            "Ignalinos m.",
            "Ignalinos r. sav.",
            "Jonavos m.",
            "Jonavos r. sav.",
            "Joniškio r. sav.",
            "Joniškio m.",
            "Jurbarko m.",
            "Jurbarko r. sav.",
            "Kaišiadorių m.",
            "Kaišiadorių r. sav.",
            "Kalvarijos m.",
            "Kalvarijos sav.",
            "Kaunas",
            "Kauno r. sav.",
            "Kazlų Rūdos sav.",
            "Kazlų Rūdos m.",
            "Kėdainių r. sav.",
            "Kėdainių m.",
            "Kelmės r. sav.",
            "Kelmės m.",
            "Klaipėda",
            "Klaipėdos r. sav.",
            "Kretingos m.",
            "Kretingos r. sav.",
            "Kupiškio r. sav.",
            "Kupiškio m.",
            "Lazdijų m.",
            "Lazdijų r. sav.",
            "Marijampolės m.",
            "Marijampolės sav.",
            "Mažeikių m.",
            "Mažeikių r. sav.",
            "Molėtų r. sav.",
            "Molėtų m.",
            "Neringos m.",
            "Pagėgių sav.",
            "Pagėgių m.",
            "Pakruojo r. sav.",
            "Pakruojo k.",
            "Pakruojo m.",
            "Palanga",
            "Panevėžio r. sav.",
            "Panevėžys",
            "Pasvalio r. sav.",
            "Pasvalio m.",
            "Plungės m.",
            "Plungės r. sav.",
            "Prienų r. sav.",
            "Prienų m.",
            "Radviliškio m.",
            "Radviliškio r. sav.",
            "Raseinių r. sav.",
            "Raseinių m.",
            "Rietavo sav.",
            "Rietavo m.",
            "Rokiškio m.",
            "Rokiškio r. sav.",
            "Šakių m.",
            "Šakių r. sav.",
            "Šalčininkų r. sav.",
            "Šalčininkų m.",
            "Šiauliai",
            "Šiaulių r. sav.",
            "Šilalės m.",
            "Šilalės r. sav.",
            "Šilutės m.",
            "Šilutės r. sav.",
            "Širvintų r. sav.",
            "Širvintų m.",
            "Skuodo m.",
            "Švenčionių m.",
            "Švenčionių r. sav.",
            "Tauragės m.",
            "Tauragės r. sav.",
            "Telšių m.",
            "Telšių r. sav.",
            "Trakų r. sav.",
            "Trakų m.",
            "Ukmergės m.",
            "Ukmergės r. sav.",
            "Utenos m.",
            "Varėnos m.",
            "Varėnos r. sav.",
            "Vilkaviškio r. sav.",
            "Vilkaviškio m.",
            "Vilniaus r. sav.",
            "Vilnius",
            "Visagino m.",
            "Zarasų r. sav.",
            "Zarasų m."
],
            index=0,
            help="Select the city where the property is located"
        )
        
        plotas = st.number_input(
            "Area (m²)",
            min_value=10.0,
            max_value=1000.0,
            value=st.session_state.property_data.get("plotas", 50.0),
            step=1.0,
            help="Total area of the property in square meters"
        )
        
        kambariu_sk = st.number_input(
            "Number of rooms",
            min_value=1,
            max_value=20,
            value=st.session_state.property_data.get("kambariu_sk", 2),
            step=1,
            help="Total number of rooms in the property"
        )
    
    with col2:
        aukstas = st.number_input(
            "Floor number",
            min_value=1,
            max_value=50,
            value=st.session_state.property_data.get("aukstas", 3),
            step=1,
            help="Floor on which the property is located"
        )
        
        aukstu_sk = st.number_input(
            "Total floors in building",
            min_value=1,
            max_value=50,
            value=st.session_state.property_data.get("aukstu_sk", 5),
            step=1,
            help="Total number of floors in the building"
        )
        
        metai = st.number_input(
            "Year built",
            min_value=1900,
            max_value=2025,
            value=st.session_state.property_data.get("metai", 2000),
            step=1,
            help="Year the building was constructed"
        )
    
    return {
        "city": city,
        "plotas": plotas,
        "kambariu_sk": kambariu_sk,
        "aukstas": aukstas,
        "aukstu_sk": aukstu_sk,
        "metai": metai
    }

def building_characteristics_inputs():
    """Create input fields for building characteristics."""
    st.subheader("Building Characteristics")
    col1, col2 = st.columns(2)
    
    with col1:
        pastato_tipas = st.selectbox(
            "Building type",
            [None, "Mūrinis", "Monolitinis", "Blokinis", "Medinis", "Karkasinis"],
            format_func=lambda x: "Not specified" if x is None else x,
            help="Type of building construction"
        )
        
        sildymas = st.selectbox(
            "Heating type",
            [None, "Centrinis", "Dujinis", "Elektrinis", "Katilinė", "Geoterminis"],
            format_func=lambda x: "Not specified" if x is None else x,
            help="Type of heating system"
        )
    
    with col2:
        irengimas = st.selectbox(
            "Interior finishing",
            [None, "Įrengtas", "Dalinė apdaila", "Neįrengtas"],
            format_func=lambda x: "Not specified" if x is None else x,
            help="Level of interior finishing"
        )
        
        pastato_energijos_suvartojimo_klase = st.selectbox(
            "Energy efficiency class",
            [None, "A++", "A+", "A", "B", "C", "D", "E", "F", "G"],
            format_func=lambda x: "Not specified" if x is None else x,
            help="Energy efficiency rating of the building"
        )
        
    return {
        "pastato_tipas": pastato_tipas,
        "sildymas": sildymas,
        "irengimas": irengimas,
        "pastato_energijos_suvartojimo_klase": pastato_energijos_suvartojimo_klase
    }

def property_features_inputs(model_type):
    """Create tabs with property feature inputs."""
    # Create tabs for better organization of features
    feature_tab1, feature_tab2, feature_tab3, feature_tab4, feature_tab5 = st.tabs([
        "🛋️ Interior Features", 
        "🏢 Building Features", 
        "🛁 Amenities",
        "🔧 Utilities", 
        "🔒 Security"
    ])
    
    # Dictionary to store all feature values
    features = {}
    
    with feature_tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Room Features**")
            features["aukstos_lubos"] = st.checkbox("High ceilings", value=False)
            features["virtuve_sujungta_su_kambariu"] = st.checkbox("Kitchen connected with room", value=False)
            features["tualetas_ir_vonia_atskirai"] = st.checkbox("Separate toilet and bathroom", value=False)
            features["butas_per_kelis_aukstus"] = st.checkbox("Multi-level apartment", value=False)
            features["butas_palepeje"] = st.checkbox("Attic apartment", value=False)
        
        with col2:
            st.markdown("**Extras**")
            features["balkonas"] = st.checkbox("Balcony", value=False)
            features["terasa"] = st.checkbox("Terrace", value=False)
            features["drabuzine"] = st.checkbox("Wardrobe", value=False)
            features["sandeliukas"] = st.checkbox("Storage room", value=False)
            features["atskiras_iejimas"] = st.checkbox("Separate entrance", value=False)
        
        with col3:
            st.markdown("**Additional Spaces**")
            features["pirtis"] = st.checkbox("Sauna", value=False)
            features["yra_palepe"] = st.checkbox("Attic", value=False)
            features["vieta_automobiliui"] = st.checkbox("Parking space", value=False)
            
            # Rental-specific options
            if model_type == "rental":
                st.markdown("**Rental Terms**")
                features["galima_deklaruoti_gyvenam"] = st.checkbox("Can declare residence", value=False)
                features["galima_su_gyv"] = st.checkbox("Can have pets", value=False)
            else:
                features["galima_deklaruoti_gyvenam"] = False
                features["galima_su_gyv"] = False
    
    with feature_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Building Features**")
            features["renovuotas_namas"] = st.checkbox("Renovated building", value=False)
            features["yra_liftas"] = st.checkbox("Elevator", value=False)
            features["uzdaras_kiemas"] = st.checkbox("Enclosed yard", value=False)
        
        with col2:
            st.markdown("**Building Improvements**")
            features["nauja_elektros_instaliacija"] = st.checkbox("New electrical installation", value=False)
            features["nauja_kanalizacija"] = st.checkbox("New sewage system", value=False)
            features["plastikiniai_vamzdziai"] = st.checkbox("Plastic pipes", value=False)
            features["rekuperacine_sistema"] = st.checkbox("Recovery system", value=False)
    
    with feature_tab3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Furniture**")
            features["su_baldais"] = st.checkbox("Furnished", value=False)
            features["virtuves_komplektas"] = st.checkbox("Kitchen set", value=False)
        
        with col2:
            st.markdown("**Kitchen Appliances**")
            features["virykle"] = st.checkbox("Stove", value=False)
            features["indaplove"] = st.checkbox("Dishwasher", value=False)
            features["skalbimo_masina"] = st.checkbox("Washing machine", value=False)
        
        with col3:
            st.markdown("**Bathroom**")
            features["duso_kabina"] = st.checkbox("Shower cabin", value=False)
            features["vonia"] = st.checkbox("Bathtub", value=False)
    
    with feature_tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Utilities**")
            features["internetas"] = st.checkbox("Internet", value=False)
            features["kabeline_televizija"] = st.checkbox("Cable TV", value=False)
            features["kondicionierius"] = st.checkbox("Air conditioning", value=False)
    
    with feature_tab5:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Security Features**")
            features["budintis_sargas"] = st.checkbox("Security guard", value=False)
            features["kodine_laiptines_spyna"] = st.checkbox("Code lock", value=False)
        
        with col2:
            st.markdown("**Security Systems**")
            features["signalizacija"] = st.checkbox("Alarm", value=False)
            features["vaizdo_kameros"] = st.checkbox("Video cameras", value=False)
            
    return features

def prepare_property_data(basic_info, building_chars, features, location, distances):
    """Prepare property data dictionary for prediction API."""
    # Start with basic info
    property_data = basic_info.copy()
    
    # Add building characteristics
    property_data.update(building_chars)
    
    # Add location data
    property_data["latitude"] = location[0]
    property_data["longitude"] = location[1]
    
    # Add distances
    property_data.update(distances)
    
    # Add binary features (convert bool to 0/1)
    for feature, value in features.items():
        property_data[feature] = 1 if value else 0
    
    return property_data

def create_prediction_button():
    """Create the predict button."""
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_button = st.button("🔮 PREDICT PRICE", type="primary", use_container_width=True)
    
    return predict_button