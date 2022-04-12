import cv2
import numpy as np
from pathlib import Path
import argparse


def parse_subfloders_content(dir, glob):
    '''
    Returns a nested list, where each row corresponds
    to a subfolder of the specified directory,
    and each element is a path to a file
    of specified extension from these directories
    '''
    
    # convert the dir argument to the Path object,
    # if a String is passed
    dir = Path(dir)
    
    # create a list where further lists of paths
    # will be stored
    subfolders_content = []
    
    # iterate through subdirectories,
    # and fill the content list
    for child in dir.iterdir():
        if child.is_dir():
            child_content = [elem for elem in child.glob(glob)]
            subfolders_content.append(child_content)
    
    return subfolders_content


def process_to_tile(path, tile_size):
    '''
    Returns a tile -- a preprocessed image,
    which is ready to be placed into a canvas.
    
    Preprocessing:
    - crop max-size square region out of the image center
    - resize to the given tile_size
    '''
    
    img = cv2.imread(str(path))
    
    # center crop
    img_center = (img.shape[0] // 2, img.shape[1] // 2)
    min_dim = min(img.shape[:2])  # the smallest among img's height or width
    cropped = img[img_center[0] - min_dim//2 : img_center[0] + min_dim//2 + 1,
                  img_center[1] - min_dim//2 : img_center[1] + min_dim//2 + 1]
    
    resized = cv2.resize(cropped, (tile_size, tile_size))
    
    return resized


def get_folder_labels(input_dir):
    folder_labels = [str(path.name) for path in input_dir.iterdir()
                     if path.is_dir()]
    
    return folder_labels


def create_mosaic(input_dir, output_path=None, glob='*.jpg',
                  num_rows=None, num_cols=None,
                  labels=None, title=None,
                  tile_size=64, padding_size=8,
                  background_color=(255, 255, 255), text_color=(0, 0, 0),
                  fontFace=0, fontScale=2, thickness=1
                  ):
    
    # create nested list:
    # every row represents a subfolder,
    # and contains a list of filepaths of files it's comprised of
    content = parse_subfloders_content(input_dir, glob)
    
    # collect text params to a dictinary
    text_params = {'fontFace': fontFace,
                   'fontScale': fontScale,
                   'thickness': thickness
                   }
    
    # derive basic canvas params
    if not num_rows:
        num_rows = len(content)
    if not num_cols:
        num_cols = max([len(subfolder) for subfolder in content])
    
    canvas_height = num_rows * (tile_size + padding_size) + padding_size
    canvas_width = num_cols * (tile_size + padding_size) + padding_size
    
    # adjust canvas height wrt title
    vertical_padding = padding_size
    if title:
        title_size = cv2.getTextSize(text=title, **text_params)
        # increment vertical padding with some
        # arbitrary (by eye) defined number of pixels
        # which just looks good wrt padding size
        vertical_padding += title_size[0][1] + tile_size // 2
    canvas_height += vertical_padding
    
    # adjust canvas width wrt labels
    horizontal_padding = padding_size
    if labels is not None:
        if not labels:
            labels = get_folder_labels(input_dir)
        max_length_label = max(labels, key=len)
        max_label_size = cv2.getTextSize(text=max_length_label, **text_params)
        # increment horizontal padding with some
        # arbitrary (by eye) defined number of pixels
        # which just looks good wrt tile size
        horizontal_padding += max_label_size[0][0] + tile_size // 2
    canvas_width += horizontal_padding
    
    # create canvas
    channels = 3
    canvas = np.full(shape=(canvas_height, canvas_width, channels),
                     fill_value=background_color, dtype=np.uint8).squeeze()
    
    # put title on canvas
    if title:
        title_y = title_size[0][1] + padding_size
        title_x = (canvas_width - title_size[0][0]) // 2  # cetner 
        cv2.putText(img=canvas, text=title, org=(title_x, title_y),
                    color=text_color, lineType=cv2.LINE_AA, **text_params)
        
    # put labels on canvas
    if labels:
        for i, label in enumerate(labels):
            label_y = vertical_padding + tile_size\
                + i * (tile_size + padding_size)
            cv2.putText(img=canvas, text=label, org=(padding_size, label_y), 
                        color=text_color, lineType=cv2.LINE_AA, **text_params)

    # fill canvas with tiles
    for i, subfolder in enumerate(content):
        if i == num_rows:
            break
        for j, path in enumerate(subfolder):
            if j == num_cols:
                break
            y = vertical_padding + i * (tile_size + padding_size)
            x = horizontal_padding + j * (tile_size + padding_size)
            canvas[y:y+tile_size,
            x:x+tile_size] = process_to_tile(path, tile_size)
    
    if output_path:
        cv2.imwrite(output_path, canvas)
    
    return canvas


def parse_arguments():
    
    # initialize a parser object
    parser = argparse.ArgumentParser()
    
    # create parser arguments
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_path', nargs='?', type=str, default=None)
    parser.add_argument('--glob', type=str, default='*.jpg')
    parser.add_argument('--num-rows', type=int, default=None)
    parser.add_argument('--num-cols', type=int, default=None)
    parser.add_argument('--labels', nargs='*', action='extend')
    parser.add_argument('--title', type=str)
    parser.add_argument('--tile-size', type=int, default=64)
    parser.add_argument('--padding-size', type=int, default=8)
    parser.add_argument('--background-color', nargs='+', type=int,
                        default=[255, 255, 255])
    parser.add_argument('--text-color', nargs='+', type=int,
                        default=[0, 0, 0])
    parser.add_argument('--fontFace', type=int, default=0)
    parser.add_argument('--fontScale', type=float, default=2)
    parser.add_argument('--thickness', type=int, default=1)

    # parse arguments from the command line
    # and convert them to a dictionary
    args = vars(parser.parse_args())
    
    return args


def main():

    args = parse_arguments()
    create_mosaic(**args)


if __name__ == '__main__':
    main()