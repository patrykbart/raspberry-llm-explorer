# Zmienne globalne do wspó³dzielenia miêdzy trasami
last_image_b64 = ""
last_prompt = ""
last_response = ""

def set_history(image, prompt, response):
    global last_image_b64, last_prompt, last_response
    last_image_b64 = image
    last_prompt = prompt
    last_response = response

def get_history():
    return {
        "image": last_image_b64,
        "prompt": last_prompt,
        "response": last_response
    }