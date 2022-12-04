from pathlib import Path

#TODO @Z:: Hardcoded path.
def encode_exif_data(image, seed, image_num, batch_number, config):
    dir_path = Path(f"./outputs/{config.prompt}")
    file_name = f"b{batch_number}-s{seed+image_num}.png"
    if not dir_path.exists():
        dir_path.mkdir()
    dest = (dir_path/file_name)

    #unicode
    prefix = bytes(b'UNICODE\x00')
    content = ""
    #hacky but whatever
    try:
        config.seed = seed
        config.generate_x_in_parallel = image_num
        config.batches = batch_number
    except:
        pass

    for cf, cv in zip(list(config), list(config.values())):
        content += cf + ": " + str(cv) + "\n"

    exif_data = image.getexif()
    user_comment_tiff = 0x9286
    exif_data[user_comment_tiff] = prefix + content.encode('utf-16le')
    image.save(dest, exif=exif_data)
    return image