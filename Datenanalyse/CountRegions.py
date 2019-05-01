import os
from xml.etree import ElementTree as ET

NOTARSURKUNDEN_XML_DIR = \
	"/home/martin/Forschungspraktikum/Testdaten/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/page/"

REGION_TYPES = {
	0: "TextRegion",
	1: "ImageRegion",
	2: "LineDrawingRegion",
	3: "GraphicRegion",
	4: "TableRegion",
	5: "ChartRegion",
	6: "SeparatorRegion",
	7: "MathsRegion",
	8: "ChemRegion",
	9: "MusicRegion",
	10: "AdvertRegion",
	11: "NoiseRegion",
	12: "UnknownRegion",
}


def main():

	region_counter = {}

	for xml_filename in os.listdir(NOTARSURKUNDEN_XML_DIR):
		xml_path = os.path.join(NOTARSURKUNDEN_XML_DIR, xml_filename)

		xml_tree = ET.parse(xml_path)
		root = xml_tree.getroot()
		ns = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}'

		for key, value in REGION_TYPES.items():
			els = root.findall("*/" + ns + value)

			if key not in region_counter:
				region_counter[key] = 0
			region_counter[key] += len(els)
			pass
		pass

	for key, value in REGION_TYPES.items():

		print(f"{value} - {region_counter[key]} times")

	pass


if __name__ == '__main__':
	main()
