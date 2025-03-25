import requests

api_key = 'AIzaSyCfLVuVigxStJXrtxfyMlBr1qAIUZPysms'
API_URL = f'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={api_key}'

headers = {
    "Content-Type": "application/json"
}

data = {
    "contents": [
        {"parts": [{"text": "Здравствуй! Проанализируй концепцию книги Код да Винчи."}]}
    ]
}


response = requests.post(API_URL, json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    generated_text = result['candidates'][0]['content']['parts'][0]['text']
    print(generated_text)
else:
    print(f"Ошибка: {response.status_code}, {response.text}")