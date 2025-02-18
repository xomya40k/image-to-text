# Image to text service

## About
The service is designed to process images containing text, followed by its recognition and conversion.

## Installation
### 1. Clone repository
```sh
git clone github.com/xomya40k/image-to-text
```
### 2. Install dependencies
```sh
pip install -r requirements.txt
```
### 3. Run
```sh
python main.py
```
#### By default, the service is available at `http://localhost:8000`

## API
### Endpoint `Upload`:
- Path: `/text_extraction/upload`
- Method: `POST` 
- Request Body (`multipart/form-data`):
    ```sh
    file: "<file_path>"
    ```
- Response Body (`application/json`):
    ```sh
    {
        "text":     "<string>",
        "status":   "<string>",
        "image":    "<base64 string>"
    }
    ```
