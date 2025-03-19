import requests
from .Model import Model

class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        self.api_keys = config["api_key_info"]["api_keys"]
        self.endpoint = config["api_key_info"]["endpoint"]

    def query(self, msg):
        def gpt4v_query(api_key, question):
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key,
            }
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "max_tokens": 300
            }
            response = requests.post(self.endpoint, headers=headers, json=payload)
            return response.json()
        
        def gpt4_evaluation(query, api_keys):
            question = query
            for api_key in api_keys[:]:
                attempts = 0
                while attempts < 10:
                    try:
                        response = gpt4v_query(api_key, question)
                        # Check for an error in the response
                        if 'error' in response:
                            if response['error'].get('code') == 'rate_limit_exceeded':
                                api_keys.remove(api_key)
                                break  # Stop processing this API key
                            else:
                                raise Exception(response['error'])

                        # Process a valid response
                        gpt4v_response = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                        return gpt4v_response
                        
                    except Exception as e:
                        print(f"Error with API Key {api_key} on attempt {attempts + 1}: {e}")
                        print(f"with question: {question}")
                        attempts += 1
                        if attempts >= 10:
                            print("Maximum retry attempts reached.")
                        continue
        
        response = gpt4_evaluation(msg, self.api_keys)
        print(response)

        return response