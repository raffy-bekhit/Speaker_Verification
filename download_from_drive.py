import requests
import sys
from data_preprocess import save_spectrogram_tisv
from zipfile import ZipFile
import os
import shutil

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
    i=60
    f = open("drive_zip_ids.txt",'r')
    ids_file = f.readlines()
    for line in ids_file:
        id, name = line.split(",")
        id = id.strip(' ')

        name = name.strip('\n')
        print(id," ",name)
        download_file_from_google_drive(id,"./audio/"+name+".zip" )
        with ZipFile("./audio/"+name+".zip",'r') as zipObj:
            zipObj.extractall("audio/"+name)
        os.remove("./audio/"+name+".zip")
        save_spectrogram_tisv(i)
        shutil.rmtree("./audio/"+name, ignore_errors=True)
        i=i+1
    f.close()
