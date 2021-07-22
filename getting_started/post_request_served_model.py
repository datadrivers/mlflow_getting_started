""" Example: Send a post request to served model endpoint on localhost
"""
import requests

url = 'http://127.0.0.1:8080/invocations'
myobj = {
  "inputs": [[
    12.7,
    3.87,
    2.4,
    23,
    101,
    2.83,
    2.55,
    0.43,
    1.95,
    2.57,
    1.19,
    3.13,
    463
  ]]
}

x = requests.post(url, json=myobj)

print(x.text)
