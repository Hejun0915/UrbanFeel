import PIL

def is_image_corrupted(file_path):
    try:
        with PIL.Image.open(file_path) as img:
            img.verify()  
        with PIL.Image.open(file_path) as img:
            img.load()  
        return False  
    except Exception as e:
        print(f"Image damaged : {file_path},Error:{e}")
        return True

def fetch_local_image(image_url: str, image_mode: str = "RGB"):
    return PIL.Image.open(image_url).convert(image_mode)

def fetch_local_image_north(image_url: str, image_mode: str = "RGB", random_percentage: float = 0.2):
    image = PIL.Image.open(image_url).convert(image_mode)

    width, height = image.size


    left_width = int(width * random_percentage) 
    right_image = image.crop((left_width, 0, width, height)) 


    left_image = image.crop((0, 0, left_width, height)) 


    combined_image = PIL.Image.new(image_mode, (width * 2 - left_width, height)) 
    combined_image.paste(left_image, (0, 0))  
    combined_image.paste(right_image, (left_width, 0))  

    return combined_image