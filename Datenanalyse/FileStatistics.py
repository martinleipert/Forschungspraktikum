import xml.etree.ElementTree as ET
import datetime
import re
import numpy
from matplotlib import pyplot as plt

"""
Martin Leipert
martin.leipert@fau.de

Statistics about the files and their age

"""

PATH_NOTARSURKUNDEN = "/home/martin/Forschungspraktikum/Testdaten/Notarsurkunden/Notarsurkunden.xml"
PATH_KEINE_NOTARSURKUNDEN = "/home/martin/Forschungspraktikum/Testdaten/KeineNotarsurkunden/KeineNotarsurkunden.xml"

TIME_STAMP_FORMAT = "%Y-%m-%d"


def main():
	notary_archiv_fonds, notary_dates_list = extract_statistical_dates(PATH_NOTARSURKUNDEN)
	no_not_archiv_fonds, no_not_dates_list = extract_statistical_dates(PATH_KEINE_NOTARSURKUNDEN)

	notary_year_histo, year_bins = calc_date_statistics(notary_dates_list)
	no_not_year_histo, year_bins = calc_date_statistics(no_not_dates_list)

	"""
	Plot the year statistics
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("Year Statistics of the Documents")
	ax.set_xlabel("Year")
	ax.set_ylabel("Occurences")
	x_ticks = year_bins[1:37:5]
	ax.set_xticks(numpy.arange(0, 38, 5))
	ax.set_xticklabels(list(map(lambda x: str(datetime.date.fromordinal(x).year), x_ticks)))
	ax.bar(numpy.arange(-0.5, 35.5, 1), no_not_year_histo[0], color="r", label="Non-Notary", bottom=notary_year_histo[0])
	ax.bar(numpy.arange(-0.5, 35.5, 1), notary_year_histo[0], color="b", label="Notary")
	ax.legend(loc=2)
	ax.grid()
	fig.savefig("Plots/Year_Occurence_Statistics.png", dpi=200)

	set_list = get_set_list(no_not_archiv_fonds + no_not_archiv_fonds)
	set_sta_notary = set_statistics(notary_archiv_fonds, set_list)
	set_sta_no_not = set_statistics(no_not_archiv_fonds, set_list)

	"""
	Plot the year statistics of both sets
	"""
	fig = plt.figure(figsize=(20, 7))
	ax = fig.add_subplot(111)
	ax.set_title("Share of the Sets")
	ax.set_xlabel("Set Name")
	ax.set_ylabel("Occurences")
	ax.set_xticks(numpy.arange(0.5, 37.5, 1))
	ax.set_xticklabels(set_list, rotation=45, va="top", ha="right", position=(0,0), fontdict={"fontsize" : 10})
	ax.bar(numpy.arange(0.5, 37.5, 1), list(set_sta_no_not.values()), color="r", label="Non-Notary",
		bottom=list(set_sta_notary.values()))
	ax.bar(numpy.arange(0.5, 37.5, 1), list(set_sta_notary.values()), color="b", label="Notary")
	ax.legend(loc=1)
	fig.savefig("Plots/Set_Statistics.png", dpi=200, bbox_inches='tight',)


# Parse an xml to find relevant statistics
def extract_statistical_dates(xml_file):

	etree = ET.parse(xml_file)
	charters = etree.getroot().findall("charter")

	# List the dates of creation
	dates_list = []

	# Archive fond -> source the document belongs to
	archiv_fonds = []

	for charter in charters:
		archiv_fond = charter.find("archivfond").text
		date_str = charter.find("date").text
		try:
			date = datetime.datetime.strptime(date_str, TIME_STAMP_FORMAT).date()
		except ValueError as e:
			# Handle the 29. Februar of 1415
			year, month, day = map(int, re.search("(\d+)-(\d+)-(\d+)", date_str).groups())

			# Handle exceptional cases like faulty dates
			if day == 29 and month == 2:
				date = datetime.date(year=year, month=month, day=day-1)
			if day == 99 or month == 99:
				date = datetime.date(year=year, month=1, day=1)

		archiv_fonds.append(archiv_fond)
		dates_list.append(date)

	return archiv_fonds, dates_list


# Calculate the statistics
def calc_date_statistics(date_lists):
	dates_as_ordinal = list(map(lambda x: x.toordinal(), date_lists))

	min_bin = 780
	max_bin = 1500

	nr_bins = (max_bin - min_bin)/20 + 1

	year_bins = list(map(lambda x: x, numpy.linspace(780, 1500, int(nr_bins))))
	year_bins = list(map(lambda x: datetime.date(year=int(x), month=1, day=1).toordinal(), year_bins))

	histo = numpy.histogram(dates_as_ordinal, bins=year_bins)
	histo = (list(map(int, histo[0])), list(map(lambda x: datetime.date.fromordinal(int(x)), histo[1])))

	return histo, year_bins


# Get the statistics of the sets
def set_statistics(all_items, set_list):

	counts = {}

	for set_name in set_list:
		counts[set_name] = all_items.count(set_name)

	return counts


def get_set_list(all_items):

	set_list = []

	for item in all_items:
		if item not in set_list:
			set_list.append(item)

	return sorted(set_list)


if __name__ == '__main__':
	main()
