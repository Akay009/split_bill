import os
import base64
import hashlib
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from rapidfuzz import fuzz
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Catalytic Converter Inventory", page_icon="🔎", layout="wide")

MONGO_URI = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "converter_inventory")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "converters")

ACCESS_KEYS = set(
    key.strip()
    for key in os.getenv("ACCESS_KEYS", "").split(",")
    if key.strip()
)

if not MONGO_URI:
    st.error("MONGO_URL is missing in your .env file.")
    st.stop()


def hash_key(key: str) -> str:
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()


def validate_access_key(key: str) -> bool:
    if not key:
        return False

    clean_key = key.strip()
    hashed = hash_key(clean_key)

    for allowed in ACCESS_KEYS:
        if allowed == clean_key:
            return True
        if allowed == f"sha256:{hashed}":
            return True

    return False


@st.cache_resource
def get_collection():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    collection.create_index([("serial_number", ASCENDING)], unique=True)
    collection.create_index([("company_name", ASCENDING)])
    collection.create_index([("price", ASCENDING)])

    return collection


def image_to_base64(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None

    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((900, 900))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(image_b64: str):
    return BytesIO(base64.b64decode(image_b64))


def normalize_text(value) -> str:
    return str(value or "").strip().lower()


def fuzzy_score(query: str, converter: dict) -> int:
    query = normalize_text(query)

    fields = [
        normalize_text(converter.get("serial_number")),
        normalize_text(converter.get("company_name")),
        normalize_text(converter.get("converter_name")),
        normalize_text(converter.get("price")),
    ]

    return max(fuzz.WRatio(query, field) for field in fields if field) if query else 0


def search_converters(collection, query: str, min_score: int, max_results: int):
    query = query.strip()

    if not query:
        docs = list(
            collection.find({}, {"image_b64": 0})
            .sort("created_at", -1)
            .limit(max_results)
        )
        return [(doc, None) for doc in docs]

    regex_filter = {
        "$or": [
            {"serial_number": {"$regex": query, "$options": "i"}},
            {"company_name": {"$regex": query, "$options": "i"}},
            {"converter_name": {"$regex": query, "$options": "i"}},
            {"price_text": {"$regex": query, "$options": "i"}},
        ]
    }

    candidates = list(collection.find(regex_filter).limit(300))

    if len(candidates) < max_results:
        candidates = list(collection.find({}).sort("created_at", -1).limit(1000))

    scored = []

    for doc in candidates:
        score = fuzzy_score(query, doc)
        if score >= min_score:
            scored.append((doc, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    return scored[:max_results]


def show_converter_card(doc: dict, score: int | None = None):
    with st.container(border=True):
        cols = st.columns([1, 3])

        with cols[0]:
            if doc.get("image_b64"):
                st.image(base64_to_image(doc["image_b64"]), use_column_width=True)
            else:
                st.caption("No image")

        with cols[1]:
            title = doc.get("converter_name") or "Unnamed converter"

            if score is not None:
                title += f" · Match {score}%"

            st.subheader(title)
            st.write(f"**Serial number:** {doc.get('serial_number', '')}")
            st.write(f"**Company:** {doc.get('company_name', '')}")
            st.write(f"**Price:** {doc.get('price', '')}")

            if doc.get("notes"):
                st.write(f"**Notes:** {doc.get('notes')}")


st.title("Catalytic Converter Inventory")
st.caption("Store and search catalytic converter records in MongoDB.")

if not ACCESS_KEYS:
    st.error("ACCESS_KEYS is missing in your .env file.")
    st.stop()

with st.sidebar:
    st.header("Access")
    access_key = st.text_input("Enter access key", type="password")

    if validate_access_key(access_key):
        st.session_state["authenticated"] = True
        st.success("Access granted")
    elif access_key:
        st.session_state["authenticated"] = False
        st.error("Invalid access key")

if not st.session_state.get("authenticated", False):
    st.info("Enter a valid access key to continue.")
    st.stop()

collection = get_collection()

add_tab, search_tab = st.tabs(["Add converter", "Search converters"])

with add_tab:
    st.header("Add converter")

    with st.form("add_converter_form", clear_on_submit=True):
        serial_number = st.text_input("Serial number *")
        converter_name = st.text_input("Converter name *")
        company_name = st.text_input("Company name *")
        price = st.number_input("Price *", min_value=0.0, step=1.0, format="%.2f")
        image = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
        notes = st.text_area("Notes")

        submitted = st.form_submit_button("Save converter")

    if submitted:
        if not serial_number or not converter_name or not company_name:
            st.error("Serial number, converter name, and company name are required.")
        else:
            doc = {
                "serial_number": serial_number.strip(),
                "converter_name": converter_name.strip(),
                "company_name": company_name.strip(),
                "price": float(price),
                "price_text": str(price),
                "image_b64": image_to_base64(image),
                "notes": notes.strip(),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            try:
                collection.insert_one(doc)
                st.success("Converter saved successfully.")
            except DuplicateKeyError:
                st.error("A converter with this serial number already exists.")

with search_tab:
    st.header("Search converters")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        query = st.text_input("Search by serial number, company, converter name, or price")

    with col2:
        min_score = st.slider("Fuzzy match", min_value=40, max_value=100, value=65)

    with col3:
        max_results = st.number_input("Max results", min_value=1, max_value=100, value=20)

    results = search_converters(collection, query, min_score, int(max_results))

    st.write(f"Found **{len(results)}** result(s).")

    for doc, score in results:
        show_converter_card(doc, score)

    if results:
        export_rows = []

        for doc, score in results:
            export_rows.append(
                {
                    "serial_number": doc.get("serial_number"),
                    "converter_name": doc.get("converter_name"),
                    "company_name": doc.get("company_name"),
                    "price": doc.get("price"),
                    "match_score": score,
                    "notes": doc.get("notes"),
                }
            )

        st.download_button(
            "Download results as CSV",
            pd.DataFrame(export_rows).to_csv(index=False),
            file_name="converter_search_results.csv",
            mime="text/csv",
        )