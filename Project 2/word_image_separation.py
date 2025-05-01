from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET
import os
import time

path_images = "../documents/binarized_images/"
path_shape_of_images = "../documents/word_images/"
path_locations = "../documents/ground-truth/locations/"
nb_words = 3726  # adjust if needed based on actual number of words

# Create word image directory if it doesn't exist
if not os.path.exists(path_shape_of_images):
    os.makedirs(path_shape_of_images)


def crop_words(list_doc_file, file_nb, word, all_coord, all_box, id_word):
    im = Image.open(path_images + list_doc_file[file_nb]).convert("RGBA")
    imArray = np.asarray(im)
    polygon = all_coord[word]
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)
    newImArray = np.empty(imArray.shape, dtype='uint8')
    newImArray[:, :, :3] = imArray[:, :, :3]
    newImArray[:, :, 3] = mask * 255
    newIm = Image.fromarray(newImArray, "RGBA")
    im_crop = newIm.crop((all_box[word][0], all_box[word][1], all_box[word][2], all_box[word][3]))

    blank_image = Image.new('RGBA', (all_box[word][2] - all_box[word][0], all_box[word][3] - all_box[word][1]), (255, 255, 255, 255))
    blank_image.paste(im_crop, (0, 0), im_crop)
    blank_image.save(path_shape_of_images + '{}.png'.format(id_word[word]))


# Crop words if not already done
if nb_words != len([name for name in os.listdir(path_shape_of_images) if os.path.isfile(os.path.join(path_shape_of_images, name))]):
    print('Words are not cropped yet. Cropping them now... (time duration: approx. 7 min)')
    start = time.time()

    list_path_svg = []
    list_path_doc = []

    # Updated range for your documents
    for i in range(305, 310):  # 305 to 309 inclusive
        list_path_svg.append(f"{path_locations}{i}.svg")
        list_path_doc.append(f"{i}b.jpg")  # matches your binarized images

    for ind, page in enumerate(list_path_svg):
        tree = ET.parse(page)
        all_coordinates_of_words = []
        all_word_box = []
        words_id = []

        for elem in tree.findall(".//*[@id]"):
            x_ax, y_ax = 10000, 10000
            width, height = 0, 0
            word_coord = []
            list_of_coord = elem.get('d').split()
            for index, element in enumerate(list_of_coord):
                if element == 'M':
                    point = (float(list_of_coord[index + 1]), float(list_of_coord[index + 2]))
                    word_coord.append(point)
                elif element == 'L':
                    point = (float(list_of_coord[index + 1]), float(list_of_coord[index + 2]))
                    word_coord.append(point)
                    if min(float(list_of_coord[index - 2]), float(list_of_coord[index + 1])) < x_ax:
                        x_ax = min(round(float(list_of_coord[index - 2])), round(float(list_of_coord[index + 1])))
                    if min(float(list_of_coord[index - 1]), float(list_of_coord[index + 2])) < y_ax:
                        y_ax = min(round(float(list_of_coord[index - 1])), round(float(list_of_coord[index + 2])))
                    if max(float(list_of_coord[index - 2]), float(list_of_coord[index + 1])) > width:
                        width = max(round(float(list_of_coord[index - 2])), round(float(list_of_coord[index + 1])))
                    if max(float(list_of_coord[index - 1]), float(list_of_coord[index + 2])) > height:
                        height = max(round(float(list_of_coord[index - 1])), round(float(list_of_coord[index + 2])))
            all_coordinates_of_words.append(word_coord)
            all_word_box.append([x_ax, y_ax, width, height])
            words_id.append(elem.get('id'))

        for i in range(len(all_coordinates_of_words)):
            crop_words(list_path_doc, ind, i, all_coordinates_of_words, all_word_box, words_id)
        print(f"✅ All words of document #{305 + ind} have been cropped.")

    print("✅ Cropping process finished in {} seconds".format(round(time.time() - start)))
else:
    print("✅ Image Cropping is already done.")
