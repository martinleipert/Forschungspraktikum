import os
from xml.etree import ElementTree as ET
import cv2
from PIL import Image, ImageDraw, ImageFilter
import numpy
from matplotlib import pyplot

REGION_TYPES = {
	"TextRegion": 0,
	"ImageRegion": 1,
	"GraphicRegion": 2
}

DIR_ROOT = "/home/martin/Forschungspraktikum/Testdaten/"

NOTARSURKUNDEN_DIR = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/"
XMLFILE_DIR = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/page"
OUT_DIR = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/advanced_segmented"
SCHEMA = http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15
Image.MAX_IMAGE_PIXELS = 933120000


def main():

	if not os.path.exists(OUT_DIR):
		os.mkdir(OUT_DIR)

	for xml_filename in os.listdir(XMLFILE_DIR):
		generate_mask_dynamically(xml_filename, XMLFILE_DIR, NOTARSURKUNDEN_DIR)
		numpy.save(os.path.join(OUT_DIR, xml_filename.rstrip(".xml")), img_array)


def generate_mask_dynamically(xml_filename, xmlfile_dir, notarsurkunden_dir):
	xml_path = os.path.join(xmlfile_dir, xml_filename)

	xml_tree = ET.parse(xml_path)
	root = xml_tree.getroot()
	ns = f'{SCHEMA}'

	jpg_filename = "%s.jpg" % xml_filename.rstrip(".xml")

	jpg_path = os.path.join(notarsurkunden_dir, jpg_filename)

	pil_image = Image.open(jpg_path)

	img_array = numpy.float32(numpy.zeros(list(pil_image.size) + [len(REGION_TYPES)]))

	for key, value in REGION_TYPES.items():
		els = root.findall("*/" + ns + key)

		for el in els:
			points = el.find(ns + 'Coords').get('points')
			fix_pts = tuple(map(lambda x: tuple(map(int, x.split(','))), points.split(' ')))

			img = Image.new('L', [pil_image.size[1], pil_image.size[0]], 0)
			ImageDraw.Draw(img).polygon(fix_pts, fill=1, outline=1, )
			mask = numpy.array(img)

			img_array[:, :, value] = numpy.logical_or(mask, img_array[:, :, value])

	return img_array


def mark_region(img_array, point_string):
	points = point_string.split()
	fixpts = []
	for pt in points:
		x, y = pt.split(',')
		fixpts.append(float(x))
		fixpts.append(float(y))
	pass

	return img_array




if __name__ == '__main__':
	main()
