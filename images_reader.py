# import requests
#
#
# class CSVReaderUsingWebApi():
#     def __init__(self, url):
#         self.url = url
#
#     def read_csv(self):
#
#         response = requests.get(self.url)
#         if response.status_code == 200:
#             content = response.content.decode('utf-8')
#
#             rows = []
#             for i, row in enumerate(reader):
#                 # we drop second column (region) we keep first column (country)
#                 row.pop(1)
#                 rows.append(row)
#             return rows
#         else:
#             raise RuntimeError("Failed to download dataset: " + str(response.status_code))