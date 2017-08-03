#Thanks to luc 
#Thought to be a preprocessing fase to learn for Thalamus
import ImageFile

# ------- get main color from image ------- #
def get_main_color(files):
    img = Image.open(files)
    colors = img.getcolors(256) #put the value depending on the number of colors
    max_occurence, most_present = 0,0
    try:
        for c in colors:
            if c[0] > max_occurence:
                (max_occurence, most_present) = c
        return most_present
    except TypeError:
        raise Exception('Too many colors in the image')
