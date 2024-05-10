import streamlit as st
import pickle

def load_model_and_features():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_map.pkl', 'rb') as f:
        feature_map = pickle.load(f)
    return model, feature_map

def encode_input(features, feature_map):
    encoded = []
    for feature, value in features.items():
        encoded.append(feature_map[feature][value])
    return encoded

def main():
    st.title("Traffic Volume Prediction")
    model, feature_map = load_model_and_features()

    # Define categories based on feature map
    categories = {feature: list(feature_map[feature].keys()) for feature in feature_map}

    features = {}
    for feature, options in categories.items():
        features[feature] = st.selectbox(f'Select {feature}', options)

    if st.button('Predict Traffic Volume'):
        encoded_features = encode_input(features, feature_map)
        prediction = model.predict([encoded_features])
        st.success(f'The predicted traffic volume level is: {prediction[0]}')

if __name__ == '__main__':
    main()
