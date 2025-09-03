from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import requests
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory user storage (for demo only; replace with DB in production)
users = {}

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Gemini API setup failed: {e}")

# Authentication

@app.route('/')
def home():
    if 'user' in session:
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('home'))
        flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash("Username already exists", "warning")
        else:
            users[username] = password
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Style Recommender Routes

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_preferences = {
        "color": request.form.get('color', '').strip(),
        "gender": request.form.get('gender', '').strip(),
        "type": request.form.get('type', '').strip(),
        "occasion": request.form.get('occasion', '').strip(),
        "style": request.form.get('style', '').strip()
    }

    raw_query = construct_query(user_preferences)
    logger.info(f"Raw query: {raw_query}")

    refined_query = refine_query_gemini(raw_query) if GEMINI_API_KEY else raw_query
    logger.info(f"Refined query: {refined_query}")

    image_urls = scrape_duckduckgo_images(refined_query)
    description = generate_description(user_preferences)

    accessories_response = None
    if 'accessory_items' in request.form:
        accessory_items = request.form.getlist('accessory_items')
        accessories_response = generate_accessories(refined_query, user_preferences['gender'], accessory_items)

    return render_template(
        'recommendation.html',
        query=refined_query,
        image_urls=image_urls,
        description=description,
        preferences=json.dumps(user_preferences),
        accessories_response=accessories_response
    )

@app.route('/ask-accessories', methods=['POST'])
def ask_accessories():
    prefs = request.form.get('preferences')
    return render_template("ask_accessories.html", preferences=prefs)

@app.route('/accessories', methods=['POST'])
def accessories():
    try:
        prefs = json.loads(request.form.get('preferences', '{}'))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse preferences JSON: {e}")
        prefs = {}

    items = request.form.getlist('items')
    outfit_desc = construct_query(prefs)
    gender = prefs.get("gender", "unisex")

    text = generate_accessories(outfit_desc, gender, items)
    accessory_images = fetch_accessory_images(items)

    return render_template(
        "accessories.html",
        outfit=outfit_desc,
        accessories=text,
        accessory_images=accessory_images
    )

# Helper functions...


# Helper functions

def construct_query(prefs):
    return " ".join(filter(None, [
        prefs.get('color'),
        prefs.get('style'),
        prefs.get('type'),
        f"for {prefs.get('occasion')}" if prefs.get('occasion') else "",
        prefs.get('gender')
    ])).strip()

def refine_query_gemini(raw_query):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"Refine the following fashion query for concise image search (max 8 keywords). Output ONLY the refined query: \"{raw_query}\""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=60, temperature=0.5),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )

        if not response.candidates:
            return raw_query

        refined = response.text.strip()
        return " ".join(refined.split()[:8]) or raw_query

    except Exception as e:
        logger.error(f"Gemini refinement failed: {e}")
        return raw_query

def generate_description(prefs):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = (
            f"Generate a 4-5 line fashion description based on the following preferences:\n"
            f"Color: {prefs.get('color', 'any')}, Gender: {prefs.get('gender', 'any')}, "
            f"Type: {prefs.get('type', 'any')}, Occasion: {prefs.get('occasion', 'any')}, Style: {prefs.get('style', 'any')}.\n"
            "Write a friendly paragraph, no bullets."
        )
        response = model.generate_content(prompt)
        return response.text.strip() if response.candidates else "A stylish look for your preferences."
    except Exception as e:
        logger.error(f"Failed to generate description: {e}")
        return "A stylish look for your preferences."

def generate_accessories(outfit_desc, gender, items):
    prompt = (
        f"Suggest fashionable matching accessories for this outfit:\n"
        f"Outfit: {outfit_desc}\n"
        f"Gender: {gender}\n"
        f"Requested Accessories: {', '.join(items)}\n\n"
        "Write a friendly paragraph that recommends stylish matching accessories. "
        "Include why they go well with the outfit. Keep it elegant and fashion-focused."
    )
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Accessory generation failed: {e}")
        return "Some matching accessories to enhance your look beautifully!"

def get_placeholder_image():
    return "https://via.placeholder.com/400x500.png?text=Image+Not+Found"

def scrape_duckduckgo_images(query, max_images=5):
    images = []
    try:
        with DDGS() as ddgs:
            results = ddgs.images(keywords=query, region="wt-wt", safesearch="moderate", layout="square", max_results=max_images * 2)
            for result in results:
                image_url = result.get("image")
                if isinstance(image_url, str) and image_url.startswith("http"):
                    images.append(image_url)
                if len(images) >= max_images:
                    break
        return images if images else [get_placeholder_image()] * max_images
    except Exception as e:
        logger.error(f"Image scraping failed: {e}")
        return [get_placeholder_image()] * max_images
    
def fetch_accessory_images(accessories_list):
    images = {}
    try:
        with DDGS() as ddgs:
            for item in accessories_list:
                results = ddgs.images(keywords=f"{item} accessory", region="wt-wt", safesearch="moderate", layout="square", max_results=1)
                for result in results:
                    image_url = result.get("image")
                    if image_url:
                        images[item] = image_url
                        break
    except Exception as e:
        logger.error(f"Accessory image fetching failed: {e}")
    return images

# Run
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("⚠️ Warning: GEMINI_API_KEY not set. Some features may not work.")
    app.run(debug=True, host='0.0.0.0', port=5000)