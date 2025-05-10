from flask import jsonify

last_image_b64 = ""
last_prompt = ""
last_response = ""

def register_status(app):
    @app.route('/status', methods=['GET'])
    def status():
        return jsonify({
            "image": last_image_b64,
            "prompt": last_prompt,
            "response": last_response
        })

def set_last_state(image_b64, prompt, response_text):
    global last_image_b64, last_prompt, last_response
    last_image_b64 = image_b64
    last_prompt = prompt
    last_response = response_text