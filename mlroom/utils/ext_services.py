import requests
from mlroom.config import WEB_API_KEY
import json
# enables to call internal services externally
#will be moved to external module, 

def upload_file(filename, server):
    """Upload a file to the specified URL."""
    api_url = f"http://{server}:8000/model/upload_model"
    headers = {'X-API-Key': WEB_API_KEY} 

    # Ensure filename is a string, not a Path object
    filename_str = str(filename)

    with open(filename, 'rb') as file:
        files = {'file': (filename_str, file)}
        response = requests.post(api_url, headers=headers, files=files)
    print(response.text)
    if response.status_code == 200:
        return 0, "ok"  # Assuming the API returns a list of UUIDs
    else:
        error_detail = response.json().get('detail', 'No additional detail')  # Extract detail
        error = f"Error uploading: {response.status_code}, Detail: {error_detail}"
        print(error)
        return -1, error



def get_archived_runners_list_by_batch_id(batch_id, server):
    """Retrieve a list of runner IDs for a given batch ID."""
    api_url = f"http://{server}:8000/archived_runners/batch/{batch_id}"
    headers = {'X-API-Key': WEB_API_KEY} 

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return 0, response.json()  # Assuming the API returns a list of UUIDs
    else:
        error_detail = response.json().get('detail', 'No additional detail')  # Extract detail
        print(f"Error fetching runnerlist of Batch {batch_id}: {response.status_code}, Detail: {error_detail}")
        return -1, []
    
def get_archived_runner_header_by_id(runner_id, server):
    """Retrieve the header data for a specific runner by ID."""
    api_url = f"http://{server}:8000/archived_runners/{runner_id}"
    headers = {'X-API-Key': WEB_API_KEY} 

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return 0, response.json()  # Assuming the response is JSON formatted
    else:
        error_detail = response.json().get('detail', 'No additional detail')  # Extract detail
        error= f"Error fetching runner with ID {runner_id}: {response.status_code}, Detail: {error_detail}"
        print(error)
        return -1, error

def get_archived_runner_detail_by_id(runner_id, server):
    """Retrieve the detail data for a specific runner by ID."""
    api_url = f"http://{server}:8000/archived_runners_detail/{runner_id}"
    headers = {'X-API-Key': WEB_API_KEY} 

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return 0, response.json()  # Assuming the response is JSON formatted
    else:
        error_detail = response.json().get('detail', 'No additional detail')  # Extract detail
        error= f"Error fetching runner detail with ID {runner_id}: {response.status_code}, Detail: {error_detail}"
        print(error)
        return -1, error

# Local debugging
# if __name__ == '__main__':
#     batch_id = "73ad1866"
#     res, val = get_archived_runners_list_by_batch_id(batch_id=batch_id)
#     print("batchrunnrs:",val)
    
#     res, val = get_archived_runner_header_by_id("5b3b9b15-75ac-43c6-8a6c-bf2b6a523faf")
#     print("runner:", val)

#     res, val = get_archived_runner_detail_by_id("5b3b9b15-75ac-43c6-8a6c-bf2b6a523faf")
#     print("runner detail:", val)
