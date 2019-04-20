from TranskribusPyClient.TranskribusPyClient.client import TranskribusClient


def main():
	login = "martin.leipert@fau.de"
	password = "66e29vgl"
	collection_id = 149734

	DL_DIR = "/home/martin/Forschungspraktikum/Testdaten/Transkribierte_Notarsurkunden/"

	transkribus_client = TranskribusClient()
	transkribus_client.auth_login(login, password)

	transkribus_client.download_collection(collection_id, DL_DIR)

	transkribus_client.auth_logout()
	pass


if __name__ == "__main__":
	main()
