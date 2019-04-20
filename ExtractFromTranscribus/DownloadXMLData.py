import urllib
import json

AUTH_URL = 'https://transkribus.eu/TrpServer/rest/auth/login'

def main():
	auth_dict = {
		'user': 'martin.leipert@fau.de',
		'pw': '66e29vgl'
	}

	user = 'martin.leipert@fau.de'
	pw = '66e29vgl'

	urlextend=f"{AUTH_URL}?user={user}&pw={pw}"

	auth_str = json.dumps(auth_dict)
	auth_bytes = bytes(auth_str, "utf-8")

	headers = {
		'Content-Type': 'application/json; charset=utf-8'
	}

	auth_request = urllib.request.Request(urlextend, method='POST')
	urllib.request.urlopen(auth_request)
	pass


if __name__ == '__main__':
	main()
