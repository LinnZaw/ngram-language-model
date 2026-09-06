import streamlit as st
import sys
import os

# ============================================================
# FIND SRC FOLDER
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from prediction import predict_bigram, predict_trigram


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="N-gram Language Model",
    page_icon="📖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("N-gram Language Model")

st.write(
    "Next-word prediction using Bigram and Trigram models."
)

st.divider()


# ============================================================
# USER INPUT
# ============================================================

text = st.text_input(
    "Enter your text",
    placeholder="Example: sherlock holmes"
)


# ============================================================
# MODEL SELECTION
# ============================================================

model = st.selectbox(
    "Select N-gram Model",
    ["Bigram", "Trigram"]
)


# ============================================================
# NUMBER OF PREDICTIONS
# ============================================================

top_n = st.slider(
    "Number of predictions",
    min_value=1,
    max_value=5,
    value=5
)


# ============================================================
# PREDICT
# ============================================================

if st.button("Predict Next Word"):

    if not text.strip():

        st.warning("Please enter some text.")

    else:

        words = text.lower().split()

        # ----------------------------------------------------
        # BIGRAM
        # ----------------------------------------------------

        if model == "Bigram":

            last_word = words[-1]

            predictions = predict_bigram(
                last_word,
                top_n=top_n
            )

        # ----------------------------------------------------
        # TRIGRAM
        # ----------------------------------------------------

        else:

            if len(words) < 2:

                st.warning(
                    "Trigram prediction requires at least two words."
                )

                st.stop()

            word1 = words[-2]
            word2 = words[-1]

            predictions = predict_trigram(
                word1,
                word2,
                top_n=top_n
            )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.subheader("Predicted Next Words")

        if not predictions:

            st.info(
                "No prediction found for the given input."
            )

        else:

            for rank, (word, probability, count) in enumerate(
                predictions,
                start=1
            ):

                st.write(
                    f"**{rank}. {word}**"
                )

                st.caption(
                    f"Probability: {probability:.6f} | "
                    f"Count: {count}"
                )