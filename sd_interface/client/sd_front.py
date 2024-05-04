import asyncio
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from fastapi import File

async def request_generated_frame(frame: Image, api_url: str, generation_config: dict, verify: bool = True): # Re implements the javascript function above
    # Send a request to the api to generate a new frame
    # compress the image before uploading. Not resize, but compress
    print(f"Requesting new frame from {api_url}")
    # load a different image
    img_buffer = BytesIO()
    frame.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    data = {
        'id': (None, str(generation_config['id'])),
        'sub_id': (None, str(generation_config['sub_id'])),
        'denoise': (None, str(generation_config['denoise'])),
        'cn_end_percent': (None, str(generation_config['cn_end_percent'])),
        'cn_strength': (None, str(generation_config['cn_strength'])),
        'landmark_type': (None, generation_config['landmark_type']),
        'theme_prompt': (None, generation_config['theme_prompt']),
        'custom_prompt': (None, generation_config['custom_prompt']),
        'checkpoint': (None, generation_config['checkpoint']),
        'mode': (None, generation_config['mode']),
        'keep_background': (None, str(generation_config['keep_background']).lower())  # Convert boolean to lowercase string
    }

    files = {
        'image': ('frame.jpg', img_buffer, 'image/png')
    }
    
    response = requests.post(api_url, data=data, files=files, verify=verify)
    response_json = response.json()
    print(response_json)
    return response_json

async def main():
    id = 1
    denoise = 0.5
    cn_end_percent = 0.75
    cn_strength = 0.75
    landmark_type = ""
    theme_prompt = ""
    custom_prompt = ""
    checkpoint = "crossroads_v10AlphaNoVae.safetensors"
    mode = "very_fast_canny"
    keep_background = True

    use_ssl = False
    # api_url = "host.zzimm.com"
    api_url = "lab"
    # api_url = "lab"
    # port = ":443"
    port = ":8080"
    endpoint = "/generate-images/"
    frame = np.zeros((512, 512, 3), np.uint8)  # RGB
    frame = Image.fromarray(frame)
    frames = 1
    generation_config = {
                'id': id,
                'sub_id': frames,
                'denoise': denoise,
                'cn_end_percent': cn_end_percent,
                'cn_strength': cn_strength,
                'landmark_type': landmark_type,
                'theme_prompt': theme_prompt,
                'custom_prompt': custom_prompt,
                'checkpoint': checkpoint,
                'mode': mode,
                'keep_background': keep_background
            }

    if use_ssl:
        protocol = "https://"
    else:
        protocol = "http://"

    await request_generated_frame(frame=frame, 
                            api_url=protocol+api_url+port+endpoint, 
                            generation_config=generation_config)
    
if __name__ == "__main__":
    # main()
    asyncio.run(main())