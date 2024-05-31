from flask import Flask, render_template, request, send_from_directory
import os
from match_nlp import find_matching_images

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        search_term = request.form['keyword']
        best_match, match_ratio, image_list = find_matching_images(search_term)
        return render_template('index.html', keyword=search_term, matched_word=best_match, ratio=match_ratio, animal_list=image_list)
    return render_template('index.html')

@app.route('/static/images/<path:image_filename>')
def serve_image(image_filename):
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory, 'album').replace('\\', '/')
    return send_from_directory(image_directory, image_filename)

if __name__ == "__main__":
    app.run(debug=True)
