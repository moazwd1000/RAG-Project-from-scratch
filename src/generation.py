import requests


def generation(question,context,url,api_key):
    payload = {'question': question, 'context':context}
    header = {"X-API-Key": api_key}
    response = requests.post(url + "/generate", json=payload, headers=header)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    response = response.json()
    return response["answer"]





