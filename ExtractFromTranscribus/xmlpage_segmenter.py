import numpy as np
import sys, os
import xml.etree.ElementTree as ET
import cv2
import argparse 


"""
By Vincent Christlein FAU
-> taken over by me Martin Leipert

Extracts patches from segmented document pages

The segmentation information is stored in xml files
"""


def main():
    parser = argparse.ArgumentParser(description="extract snippets from page")

    # Input options
    parser.add_argument('xmlfile')

    # Region type of the sheme e.g. GraphicRegion, TextRegion
    parser.add_argument('type', help='A Region Type of the sheme')
    parser.add_argument('--imgfolder', '-i',
                        help='input image folder')
    # Suffix of the input image with point - e.g. 'jpg'
    parser.add_argument('--suffix', '-s', default='tif',
                        help='suffix of imagename')

    # Output options
    parser.add_argument('--labelfile', '-l',
                        help='the output labelfile')
    parser.add_argument('--outfolder', '-o',
                        help='output folder')
    parser.add_argument('--binarize', action='store_true',
                        help='load element, mask it and binarize the inner'
                             ' part')
    parser.add_argument('--output', nargs='+', choices=['mask', 'image',
                                                        'snippet'],
                        help='output-types')
    args = parser.parse_args()

    xml_file = args.xmlfile
    region_type = args.type
    img_folder = args.imgfolder
    suffix = args.suffix
    label_file = args.labelfile
    out_folder = args.outfolder
    output = args.output
    binarize = args.binarize

    create_UNet_training_images(xml_file, region_type, img_folder, suffix, label_file, out_folder, output, binarize)


def create_UNet_training_images(xml_file, region_type, img_folder, suffix, label_file, out_folder, output, binarize):

    out_filename = None

    # Parse the XML File
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Use this Schema
    ns = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}'

    els = root.findall('*/' + ns + region_type)
    print(els)

    if len(els) == 0:
        print('no element found -> try to go into TextRegion')
        elss = root.find(ns + 'Page')
        for el in elss.iter(ns + 'TextRegion'):
            for ele in el.iter(ns + region_type):
                els.append(ele)
        if len(els) == 0:
            print('still no element found')
            raise Exception(
                f"No label found for {xml_file}"
            )

    # Get an imagename for input and read the image
    img_basename = os.path.splitext(os.path.basename(xml_file))[0]
    img_filename = os.path.join(img_folder, img_basename) + suffix
    img = cv2.imread(img_filename)
    print('image type:', img.dtype)

    # Exit if the image is of nonetype
    if img is None:
        raise ValueError('cannot read image: {}'.format(img_filename))

    # Exit if there is no output folder
    if not os.path.exists(out_folder):
        mkdir_p(out_folder)

    # Label for the snippet
    label_base = os.path.splitext(label_file)[0]
    labelfile_snip = open(label_base + '_snippet.txt', 'a+')

    out_mask = out_image = None
    if 'mask' in output:
        out_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    if 'image' in output:
        if binarize:
            out_image = np.ones((img.shape[0], img.shape[1]), np.uint8) * 255
        else:
            out_image = np.ones(img.shape, np.uint8) * 255

    for element in els:
        print('.')
        idd = element.get('id')
        print(idd)
        points = element.find(ns + 'Coords').get('points')
        print(points)
        if 'mask' in output or 'image' in output:
            snippet, out_mask, out_image = cut_snippet(img, points, True, mark_matrix=out_mask,
                                                       out_image=out_image, binarize=binarize)
        else:
            snippet, _, _ = cut_snippet(img, points, binarize=binarize)

        if 'snippet' in output:
            snippet_filename = os.path.join(out_folder, img_basename + '_' + idd + '.png')
            print('write {}'.format(snippet_filename))
            cv2.imwrite(snippet_filename, snippet)

            labelfile_snip.write('{} {}\n'.format(img_basename + '_' + idd, img_basename))

    if 'mask' in output:
        out_filename = os.path.join(out_folder, img_basename + '_mask.png')
        print('write {}'.format(out_filename))
        cv2.imwrite(out_filename, 255 * out_mask)
    if 'image' in output:
        out_filename = os.path.join(out_folder, img_basename + '_masked.png')
        print('write {}'.format(out_filename))
        cv2.imwrite(out_filename, out_image)

    labelfile_snip.close()

    return out_filename


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise


def cut_snippet(img, point_string, 
                use_mask=False, 
                mark_matrix=None,
                out_image=None,
                binarize=False):
    # split string and round to nearest int
    points = point_string.split()
    fixpts = []
    for pt in points:
        x,y = pt.split(',')
        fixpts.append(float(x))
        fixpts.append(float(y))
#    fixpts = np.array(np.rint( np.array(map(float, point_string.split(','))) ),
#                             np.int32)
    fixpts = np.array(fixpts)
    if fixpts[fixpts < 0].size > 0:
        print('WARNING: negative pts')
        fixpts[ fixpts < 0] = 0

    if len(fixpts) == 0:
        print("WARNING: fixpts-length == 0")
        return None
    if len(fixpts) % 2 != 0:
        print("WARNING: odd number of fixpoints, can't build points")
        return None
    # bring points in the correct format
    fixpts = np.reshape(fixpts, (-1,2))

    # note: cv2.boundingRect wants this format: [[[1,2]],[[3,5]],...] and as numpy-array
    # note: maybe the format [[[1,2],[3,5],..]] also does work
    # what a fuckup?! -> probably comes from internally handling this as matrix
    fixpts_cv = np.array([[x] for x in fixpts.tolist()], dtype=np.int32)
    x,y,width,height = cv2.boundingRect(fixpts_cv)
    bbox = [x,y,width,height]
    print(x,y,width,height)
    if use_mask == True:
        if mark_matrix is not None:
            cv2.fillConvexPoly(mark_matrix, fixpts_cv, 
                         #np.array([[[fp] for fp in fixpts.tolist()]], np.int32), 
                               (1,1,1))
        snippet = img[y:y+height, x:x+width]
        if binarize:
            if snippet.ndim == 3:                
                snippet = np.mean(snippet, axis=2).astype(np.uint8)
#            import pyvole
#            snippet_out = np.zeros(snippet.shape, np.uint8)
#            snippet = pyvole.puhma_preprocess.binarizeSu(snippet) #, snippet_out, False)
            snippet = cv2.GaussianBlur(snippet,(5,5),0,0)            
            _, snippet = cv2.threshold(snippet, 125, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        # relate fixpoints to origin
        fixpts -= np.array([x,y], np.int32)
        # This seems to have changed:
        ## note: here we need another [] around the points
        #fixpts_cv = np.array([[[fp] for fp in fixpts.tolist()]], np.int32)
        fixpts_cv = np.array([fixpts], np.int32)  
        maski = np.zeros((snippet.shape[0],snippet.shape[1]), np.uint8)
        cv2.fillConvexPoly(maski, fixpts_cv, (1,1,1))
        snippet[maski == 0] = 255                    
        if out_image is not None:
            out_image[y:y+height, x:x+width] = snippet
    else: 
        snippet = img[y:y+height, x:x+width]
        if mark_matrix is not None:
            cv2.rectangle(mark_matrix, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (1,1,1), cv2.cv.CV_FILLED)

    return snippet, mark_matrix, out_image


if __name__ == '__main__':
    main()
