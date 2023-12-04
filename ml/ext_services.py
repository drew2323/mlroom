import requests
from v2realbot.config import WEB_API_KEY
# enables to call internal services externally
#will be moved to external module, 

SERVER = "0.0.0.0"

def get_archived_runners_list_by_batch_id(batch_id):
    """Retrieve a list of runner IDs for a given batch ID."""
    api_url = f"http://{SERVER}:8000/archived_runners/batch/{batch_id}"
    headers = {'X-API-Key': WEB_API_KEY} 

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return response.json()  # Assuming the API returns a list of UUIDs
    else:
        error_detail = response.json().get('detail', 'No additional detail')  # Extract detail
        print(f"Error fetching runnerlist of Batch {batch_id}: {response.status_code}, Detail: {error_detail}")
        return []
    
def get_archived_runner_header_by_id(runner_id):
    """Retrieve the header data for a specific runner by ID."""
    api_url = f"http://{SERVER}:8000/archived_runners/{runner_id}"
    headers = {'X-API-Key': WEB_API_KEY} 

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return response.json()  # Assuming the response is JSON formatted
    else:
        error_detail = response.json().get('detail', 'No additional detail')  # Extract detail
        print(f"Error fetching runner with ID {runner_id}: {response.status_code}, Detail: {error_detail}")
        return None


# Local debugging
if __name__ == '__main__':
    batch_id = "73ad1866"
    val = get_archived_runners_list_by_batch_id(batch_id=batch_id)
    print("batchrunnrs:",val)
    
    val = get_archived_runner_header_by_id("5b3b9b15-75ac-43c6-8a6c-bf2b6a523faf")
    print("runner:", val)
