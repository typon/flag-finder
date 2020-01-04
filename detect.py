import numpy as np
import pickle
import pudb
import glob
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass(frozen=True)
class Pixel:
    red: int
    green: int
    blue: int

def get_pixel_counts(flagfile):
    im = Image.open(flagfile)
    im = im.convert("RGB")
    px = im.load()
    pixel_counts = Counter()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            r, g, b = px[x, y]
            p = Pixel(red=r, green=g, blue=b)
            pixel_counts[p] += 1

    total_pixels = im.size[0] * im.size[1]
    return pixel_counts, total_pixels



def gather_info():
    flags_dir = Path('flags-cia')
    all_pixel_counts = {}
    index = 0
    for flagfile in glob.glob(f"{flags_dir}/*"):
        pixel_counts, total_pixels  = get_pixel_counts(flagfile)
        all_pixel_counts[flagfile] = (pixel_counts, total_pixels)

    with open('all_pixel_counts-cia.pickle', 'wb') as f:
        pickle.dump(all_pixel_counts, f, pickle.HIGHEST_PROTOCOL)
    return all_pixel_counts

def closest_point(point, points):
    point = np.asarray(point)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)

def convert_to_vec(pixel_counts, total_pixels):
    final_vec = []
    weight = 0.8
    for pixel, count in pixel_counts.most_common(5):
        final_vec.append(pixel.red * weight)
        final_vec.append(pixel.green * weight)
        final_vec.append(pixel.blue * weight)
        weight = weight * 0.2
        final_vec.append(float(count/total_pixels))
    if len(final_vec) < (5 * 4):
        final_vec.extend([0] * ((5 * 4) - len(final_vec)))

    return final_vec

def id_flags(all_pixel_counts):
    target_flags = ["flag_1", "flag_2", "flag_3"]
    target_pixel_counts = {}
    for target_flagfile in target_flags:
        pixel_counts, total_pixels = get_pixel_counts(f"{target_flagfile}.png")
        target_pixel_counts[target_flagfile] = (pixel_counts, total_pixels)

    all_pixel_count_items = list(all_pixel_counts.items())
    candidate_points = np.asarray([convert_to_vec(p[0], p[1]) for p in all_pixel_counts.values()])
    for target_flagfile, (pixel_counts, total_pixels) in target_pixel_counts.items():
        target_point = convert_to_vec(pixel_counts, total_pixels)
        closest_index = closest_point(target_point, candidate_points)
        closest_flag, (closest_pixel_counts, closest_total_pixels) = all_pixel_count_items[closest_index]
        print(closest_flag)
        print(closest_pixel_counts.most_common(5))
        print(pixel_counts.most_common(5))



if __name__ == "__main__":
    all_pixel_counts = gather_info()
    # with open('all_pixel_counts.pickle', 'rb') as f:
    #     all_pixel_counts = pickle.load(f)
    id_flags(all_pixel_counts)
