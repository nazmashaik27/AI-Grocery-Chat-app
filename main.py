import streamlit as st
from openai import OpenAI
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import pandas as pd

# === Load environment variables ===
load_dotenv()

# === Clients ===
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# === File paths ===
MEMORY_FILE = "grocery_chat_memory.json"
RECEIPTS_CSV = "receipts/receipt_data.csv"
os.makedirs("receipts", exist_ok=True)  # directory for CSV storage

# === Memory Utilities ===
def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def store_message(user_id, role, content, name=None):
    memory = load_memory()
    msg = {"timestamp": datetime.now().isoformat(), "role": role, "content": content}
    if name:
        msg["name"] = name
    memory.setdefault(user_id, []).append(msg)
    save_memory(memory)

def get_chat_history(user_id):
    return load_memory().get(user_id, [])

# === Grocery Functions ===
def add_items(user_id, items_text):
    items = [i.strip() for i in items_text.replace(" and ", ",").split(",") if i.strip()]
    for item in items:
        store_message(user_id, "function", f"✅ Added: {item}", name="add_items")
    return f"✅ Added items: {', '.join(items)}"

def delete_item(user_id, item):
    mem = load_memory()
    updated = False
    new_msgs = []
    for m in mem.get(user_id, []):
        if m["role"] == "function" and m["content"].startswith("✅ Added:") and item.lower() in m["content"].lower():
            updated = True
            continue
        new_msgs.append(m)
    if updated:
        mem[user_id] = new_msgs
        save_memory(mem)
        return f"🗑️ Deleted: {item}"
    return f"❌ '{item}' not found."

def update_item(user_id, old_item, new_item):
    mem = load_memory()
    updated = False
    for m in mem.get(user_id, []):
        if m["role"] == "function" and m["content"].startswith("✅ Added:") and old_item.lower() in m["content"].lower():
            m["content"] = f"✅ Added: {new_item}"
            updated = True
            break
    if updated:
        save_memory(mem)
        return f"🔁 Updated: {old_item} → {new_item}"
    return f"❌ '{old_item}' not found."

def show_list(user_id):
    items = [m["content"].replace("✅ Added:", "").strip() for m in get_chat_history(user_id)
             if m["role"] == "function" and m["content"].startswith("✅ Added:")]
    if not items:
        return "🧾 Your grocery list is empty."
    return "🧾 Your current grocery list:\n" + "\n".join(f"- {i}" for i in items)

# === Receipt Functions ===
def extract_image_data(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()
    # Call Gemini OCR
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            "Provide only JSON: store_name, date (YYYY-MM-DD), total_amount (float), items (list of {item_name, quantity, total_price})",
            types.Part.from_bytes(data=data, mime_type="image/jpeg")
        ]
    )
    # Extract plain text JSON from response
    raw = response.text
    clear = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clear)
    except json.JSONDecodeError:
        st.error("Failed to parse receipt JSON")
        print("Raw OCR response:", clear)
        return {}
    except json.JSONDecodeError:
        st.error("Failed to parse receipt JSON")
        print("Raw OCR response:", clear)
        return {}

def save_receipt_data(data):
    # Flatten receipt data into CSV rows
    rows = []
    store_name = data.get('store_name')
    date = data.get('date')
    total_amount = data.get('total_amount')
    for item in data.get('items', []):
        rows.append({
            'store_name': store_name,
            'date': date,
            'total_amount': total_amount,
            'item_name': item.get('item_name'),
            'quantity': item.get('quantity', 1),
            'total_price': item.get('total_price', 0)
        })
    df_rows = pd.DataFrame(rows)
    # Append to CSV
    if not os.path.exists(RECEIPTS_CSV):
        df_rows.to_csv(RECEIPTS_CSV, index=False)
    else:
        df_rows.to_csv(RECEIPTS_CSV, mode='a', header=False, index=False)

# === Streamlit App ===
st.set_page_config(page_title="Grocery Assistant")
choice = st.sidebar.selectbox("Page", ["Assistant", "Dashboard"])

if choice == "Assistant":
    st.title("🛒 Grocery Assistant")
    user_id = st.text_input("User ID")
    if user_id:
        for m in get_chat_history(user_id):
            with st.chat_message(m["role"]): st.markdown(m["content"])
        prompt = st.chat_input("What would you like to do?")
        if prompt:
            store_message(user_id, "user", prompt)
            msgs = [{"role": "system", "content": "You are a grocery assistant."}]
            msgs.extend([{"role": m["role"], "content": m["content"]} for m in get_chat_history(user_id)])
            msgs.append({"role": "user", "content": prompt})
            resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
            reply = resp.choices[0].message.content
            store_message(user_id, "assistant", reply)
            with st.chat_message("assistant"): st.markdown(reply)
        up = st.file_uploader("Upload receipt", type=["jpg", "png"] )
        if up:
            data = extract_image_data(up.read())
            save_receipt_data(data)
            st.success("Receipt saved.")
elif choice == "Dashboard":
    st.title("📊 Analytics Dashboard")
    # Load CSV data
    if os.path.exists(RECEIPTS_CSV) and os.path.getsize(RECEIPTS_CSV) > 0:
        df = pd.read_csv(RECEIPTS_CSV, parse_dates=['date'])
        df['month'] = df['date'].dt.to_period('M')
        # Filters
        months = sorted(df['month'].astype(str).unique())
        stores = sorted(df['store_name'].dropna().unique())
        items = sorted(df['item_name'].dropna().unique())
        sel_months = st.sidebar.multiselect("Filter by Month", months, months)
        sel_stores = st.sidebar.multiselect("Filter by Store", stores, stores)
        sel_items = st.sidebar.multiselect("Filter by Item", items, items)
        filtered = df[
            df['month'].astype(str).isin(sel_months) &
            df['store_name'].isin(sel_stores) &
            df['item_name'].isin(sel_items)
        ]
        st.subheader("Item Spending")
        st.dataframe(filtered[['date','store_name','item_name','quantity','total_price']])
        st.subheader("Monthly Spend")
        st.bar_chart(filtered.groupby('month')['total_price'].sum())
        st.subheader("Store Spend")
        st.bar_chart(filtered.groupby('store_name')['total_price'].sum())
        st.subheader("Item Trend")
        st.line_chart(filtered.groupby(['month','item_name'])['total_price'].sum().unstack(fill_value=0))
    else:
        st.info("No receipt data available. Upload receipts via the Assistant page.")
else:
    st.info("Choose a page to get started.")
    recs = []
    # Filter receipts with items
    recs_with_items = [r for r in recs if isinstance(r.get('items'), list) and r.get('items')]
    if not recs_with_items:
        st.info("No valid receipt items found.")
    else:
        df = pd.json_normalize(recs_with_items, record_path='items', meta=['store_name','date','total_amount'], errors='ignore')
        if df.empty:
            st.info("No valid item records found.")
        else:
            # Convert types
            if 'total_price' in df:
                df['total_price'] = pd.to_numeric(df['total_price'], errors='coerce').fillna(0)
            else:
                df['total_price'] = 0
            if 'quantity' in df:
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(1).astype(int)
            else:
                df['quantity'] = 1
            df['month'] = pd.to_datetime(df['date'], errors='coerce').dt.to_period('M')
            # Filters
            months = sorted(df['month'].astype(str).unique())
            stores = sorted(df['store_name'].dropna().unique())
            items = sorted(df['item_name'].dropna().unique())
            sel_m = st.sidebar.multiselect("Filter by Month", months, months)
            sel_s = st.sidebar.multiselect("Filter by Store", stores, stores)
            sel_i = st.sidebar.multiselect("Filter by Item", items, items)
            filtered = df[df['month'].astype(str).isin(sel_m) & df['store_name'].isin(sel_s) & df['item_name'].isin(sel_i)]
            st.subheader("Item Spending")
            st.dataframe(filtered[['date','store_name','item_name','quantity','total_price']])
            st.subheader("Monthly Spend")
            st.bar_chart(filtered.groupby('month')['total_price'].sum())
            st.subheader("Store Spend")
            st.bar_chart(filtered.groupby('store_name')['total_price'].sum())
            st.subheader("Item Trend")
            st.line_chart(filtered.groupby(['month','item_name'])['total_price'].sum().unstack(fill_value=0))
