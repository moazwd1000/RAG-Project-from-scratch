from urllib import response

import requests
class ModelServerError(Exception):
    """Custom exception for model server errors."""
    pass
    
    


def generation(question,context,url,api_key):
    payload = {'question': question, 'context':context}
    header = {"X-API-Key": api_key}
    try:
        response = requests.post(url + "/generate", json=payload, headers=header,timeout=60)
        content_type = response.headers.get("content-type", "")
    except requests.exceptions.RequestException as e:
        raise ModelServerError("Model server unavailable")
    if response.status_code >= 500:
        raise ModelServerError("Model server unavailable")
    if response.status_code == 404 and "text/html" in content_type:
        raise ModelServerError("Model server unavailable")
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    response = response.json()
    return response["answer"]





