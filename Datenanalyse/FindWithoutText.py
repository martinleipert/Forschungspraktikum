import os

from xml.etree import ElementTree as ET

REGION_TYPES = {
	"Background": 0,
	"TextRegion": 1,
	"ImageRegion": 2,
	"GraphicRegion": 3
}


SCHEMA = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

NOTARY_DOCS = "/home/martin/Forschungspraktikum/Testdaten/Notarsurkunden/notarsurkunden_mom/page/"

os.chdir(NOTARY_DOCS)

no_text = []

for file in os.listdir("."):

	try:
		xml_tree = ET.parse(file)
	except Exception as e:
		pass

	root = xml_tree.getroot()
	ns = '{' + SCHEMA + '}'

	els = root.findall("*/" + ns + "TextRegion")

	if len(els) == 0:
		no_text.append(file)

no_text = sorted(no_text, key=lambda x: int(x.split("_")[0]))

print(f"Items without text No. {len(no_text)}")

for item in no_text:
	print(item)
