import asyncio
from rnet import Impersonate, Client


async def main():
    client = Client(impersonate=Impersonate.Firefox136)
    resp = await client.get("https://tls.browserleaks.com")
    print("Status Code: ", resp.status_code)
    print("Version: ", resp.version)
    print("Response URL: ", resp.url)
    print("Headers: ", resp.headers)
    print("Cookies: ", resp.cookies)
    print("Encoding: ", resp.encoding)
    print("Content-Length: ", resp.content_length)
    print("Remote Address: ", resp.remote_addr)
    print(await resp.text())


if __name__ == "__main__":
    asyncio.run(main())