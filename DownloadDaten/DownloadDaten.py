import urllib
import xml.etree.ElementTree as ET
import os

# Martin Leipert
# martin.leipert@fau.de
# 08.02.2019

# Download files annotated in an xml file from the internet
# Originally to download ~31000 medieval documents

# Hardcoded arguments for downloading
# Where are the file urls denoted
XML_FILE = "/home/martin/Forschungspraktikum/Testdaten/KeineNotarsurkunden/KeineNotarsurkunden.xml"
# Where to store the dara
STORAGE = "/home/martin/Forschungspraktikum/Testdaten/KeineNotarsurkunden/"


# Main method which gets called when executing the file
def main():

	# Count the skipped files
	skip_counter = 0

	# Extract the urls from an xml file
	# Returns a list consisting of tuples of the url of the page and of the image
	urls = get_the_urls_from_xml(XML_FILE)
	nr_urls = len(urls)

	# Iterate over the urls collected and store the files
	for counter, (url, img_url) in enumerate(urls):

		# Get the filename -> Last part of the url
		filename = img_url.split("/")[-1]

		# Store it in the directory previously defined
		where2store = os.path.join(STORAGE, filename)

		# Continue if the file was already downloaded
		if os.path.exists(where2store):
			continue

		# Try catch -> If an error occurs the file is skipped
		try:
			urllib.request.urlretrieve(img_url, where2store)
			print("%i - Downloaded %s" % (counter, filename))

		# In case the HTTP request fails
		except urllib.error.HTTPError as ex:
			skip_counter += + 1

			Warning("%s skipped - Error retrieving file:\nCode: %s - Message: %s" % (filename, ex.code, ex.msg))

			print("%s skipped - Error retrieving file:\nCode: %s - Message: %s" % (filename, ex.code, ex.msg))

		# In case the url does not exist
		except urllib.error.URLError as ex:
			skip_counter += + 1

			Warning("%s skipped - Error calling url:\nCode: %s - Message: %s" % (filename, ex.code, ex.msg))

			print("%s skipped - Error calling url:\nCode: %s - Message: %s" % (filename, ex.code, ex.msg))

	print("Finished: %i / %i skipped" % (skip_counter, nr_urls))


# Collects the urls from the file
def get_the_urls_from_xml(xml_file):

	# Parse the file with element tree
	xml_tree = ET.parse(xml_file)

	# Collect the elements containg the urls
	charters = xml_tree.getiterator("charter")

	urls = []

	# Iterate over the collected charters and store the urls to the pages and to the images in a tuple
	for charter in charters:
		url_elem = charter.find("url")
		url = url_elem.text

		image_file_elem = charter.find("imageFile")
		image_file = image_file_elem.text

		urls.append((url, image_file))
		pass

	return urls

if __name__ == '__main__':
	main()
