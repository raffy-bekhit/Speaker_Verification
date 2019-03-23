import requests
import sys
from data_preprocess import save_spectrogram_tisv
from zipfile import ZipFile


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":

    f = open("drive_zip_ids.txt",'r')
    ids_file = f.readlines()
    for line in ids_file:
        id, name = line.split(",")
        name = name.strip('\n')
        download_file_from_google_drive(id,"./audio/"+name+".zip" )
        with ZipFile("./audio/"+name+".zip",'r') as zipObj:
            zipObj.extractall(path="audio/"+name)
        save_spectrogram_tisv()
        os.remove(name+".zip")
        os.rmdir("./audio/"+"name")
