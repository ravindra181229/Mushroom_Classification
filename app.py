import pandas as pd
import numpy as np
import pickle
import streamlit as st
import logging as lg

try:
    lg.basicConfig(filename="logfile.log", level=lg.INFO, format='%(asctime)s %(name)s %(message)s')

    model = open("model1.pkl", "rb")
    model = pickle.load(model)

    encode = open("encode1.pkl", "rb")
    encode = pickle.load(encode)

    st.title("MUSHROOM CLASSIFICATION PREDICTION APPLICATION")
    st.write("### Below provided 23 species of mushroom , to predict whether a mushroom is edible or poisonous select one feature in every bar ")

    Cap_shape = st.selectbox("CAP_SHAPE", ["", "bell", "conical", "convex", "flat", "knobbed", "sunken"])

    Cap_surface = st.selectbox("CAP_SURFACE", ["", "fibrous", "grooves", "scaly", "smooth"])
    Cap_color = st.selectbox("CAP_COLOR",
                             ["", "brown", "buff", "cinnamon", "gray", "green", "pink", "purple", "red", "white",
                              "yellow"])
    Bruises = st.selectbox("BRUISES", ["", "no", "bruises"])
    Odor = st.selectbox("ODOR",
                        ["", "almond", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])
    Gill_attachment = st.selectbox("GILL_ATTACHMENT", ["", "attached", "free"])
    Gill_spacing = st.selectbox("GILL_SPACING", ["", "close", "crowded"])
    Gill_size = st.selectbox("GILL_SIZE", ["", "broad", "narrow"])
    Gill_color = st.selectbox("GILL_COLOR",
                              ["", "black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple",
                               "red", "white", "yellow"])

    Stalk_shape = st.selectbox("STALK_SHAPE", ["", "enlarging", "tapering"])

    Stalk_root = st.selectbox("STALK_ROOT", ["", "bulbous", "club", "equal", "missing"])

    Stalk_surface_above_ring = st.selectbox("STALK_SURFACE_ABOVE_RING", ["", "fibrous", "scaly", "silky", "smooth"])

    Stalk_surface_below_ring = st.selectbox("STALK_SURFACE_BELOW_RING", ["", "fibrous", "scaly", "silky", "smooth"])

    Stalk_color_above_ring = st.selectbox("STALK_COLOR_ABOVE_RING",
                                          ["", "brown", "buff", "cinnamon", "gray", "orange", "pink", "red",
                                           "white", "yellow",])

    Stalk_color_below_ring = st.selectbox("STALK_COLOR_BELOW_RING",
                                          ["", "brown", "buff", "cinnamon", "gray", "orange", "pink", "red",
                                           "red", "white", "yellow",])

    Veil_type = st.selectbox("VEIL_Type", ["", "partial"])

    Veil_color = st.selectbox("VEIL_COLOR", ["", "brown", "orange", "white", "yellow"])

    Ring_number = st.selectbox("RING_NUMBER", ["", "None", "one", "two"])

    Ring_type = st.selectbox("RING_TYPE", ["", "evanescent", "flaring", "large", "pendant", "none"])

    Spore_print_color = st.selectbox("SPORE_PRINT_COLOR",["", 'black', 'brown', 'buff', 'chocolate', 'green', 'orange',
                                                          'purple', 'white', 'yellow',])

    Population = st.selectbox("POPULATION",
                              ["", "abundant", "clustered", "numerous", "scattered", "several", "solitary"])

    Habitat = st.selectbox("HABITAT ", ["", "grasses", "leaves", "meadows", "paths", "urban", "waste", "woods"])

    data = [[Cap_shape, Cap_surface, Cap_color, Bruises, Odor, Gill_attachment, Gill_spacing, Gill_size, Gill_color,
             Stalk_shape, Stalk_root, Stalk_surface_above_ring, Stalk_surface_below_ring, Stalk_color_above_ring,
             Stalk_color_below_ring, Veil_type, Veil_color, Ring_number, Ring_type, Spore_print_color, Population,
             Habitat]]

    ok = st.button("PREDICT")
    if ok:
        if "" not in data[0]:
            x_encode = encode.transform(data)

            print(x_encode)

            pred = model.predict(x_encode)

            if pred[0] == "edible":
                st.title("MUSHROOM IS  EDIBLE")
                st.balloons()
            else:
                st.title("MUSHROOM is POISONOUS")


except Exception as e:
    print("Check logs for errors")
    lg.error("error occured")
    lg.exception(e)