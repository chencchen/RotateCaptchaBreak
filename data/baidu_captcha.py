import time
import json
import urllib.parse

import requests

cookies = {
    'BIDUPSID': '0149403893FB17D9868DDC00E811FCE7',
    'PSTM': '1615095665',
    'BAIDUID': '0149403893FB17D9C00F7D3EFEEAD014:FG=1',
    'delPer': '0',
    'PSINO': '2',
    'H_PS_PSSID': '33257_33273_31660_33594_33570_33591_26350_33265',
    'BA_HECTOR': '040g2l8k2g0l218k311g48prj0r',
    'BDORZ': 'B490B5EBF6F3CD402E515D22BCDA1598',
    'BAIDUID_BFESS': '0149403893FB17D9C00F7D3EFEEAD014:FG=1',
    'HOSUPPORT': '1',
    'HOSUPPORT_BFESS': '1',
    'pplogid': '6942ne4CqAM25cfC4EssHF4KPYKQwU5eZfkq2hxkeydl8lc3jlmZya3gKkwSeRGtnrmryRCGksGhdH6vvVel%2FMHsiFWkNBqc0k%2B5oLijPesWvVA%3D',
    'pplogid_BFESS': '6942ne4CqAM25cfC4EssHF4KPYKQwU5eZfkq2hxkeydl8lc3jlmZya3gKkwSeRGtnrmryRCGksGhdH6vvVel%2FMHsiFWkNBqc0k%2B5oLijPesWvVA%3D',
    'UBI': 'fi_PncwhpxZ%7ETaJcwoQzv%7Etk1GbYFHnDEF1qmAF2zK6dHasFcbPclSCm%7En-w%7ENK7VbkDIllyhZUKHBMdm5g',
    'UBI_BFESS': 'fi_PncwhpxZ%7ETaJcwoQzv%7Etk1GbYFHnDEF1qmAF2zK6dHasFcbPclSCm%7En-w%7ENK7VbkDIllyhZUKHBMdm5g',
    'logTraceID': 'dc5efb79487f49534f9e47dfe497392b33c9992e86bd78d6f0',
}

headers = {
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'sec-ch-ua': '"Chromium";v="88", "Google Chrome";v="88", ";Not A Brand";v="99"',
    'DNT': '1',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
    'Accept': '*/*',
    'Sec-Fetch-Site': 'same-site',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Dest': 'script',
    'Referer': 'https://www.baidu.com/s?rsv_idx=1&wd=31%E7%9C%81%E6%96%B0%E5%A2%9E%E7%A1%AE%E8%AF%8A13%E4%BE%8B+%E5%9D%87%E4%B8%BA%E5%A2%83%E5%A4%96%E8%BE%93%E5%85%A5&fenlei=256&ie=utf-8&rsv_cq=np.random.choice+%E4%B8%8D%E9%87%8D%E5%A4%8D&rsv_dl=0_right_fyb_pchot_20811_01&rsv_pq=c0b53cdc0005af92&oq=np.random.choice+%E4%B8%8D%E9%87%8D%E5%A4%8D&rsv_t=2452p17G6e88Hpj%2FkNppuwT%2FFjr8KeLJKT4KqqeSLqr7MhD7HbIYjtM9KVc&rsf=84b938b812815a59afcce7cc4e641b1d_1_15_8&rqid=c0b53cdc0005af92',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

params = (
    ('ak', '1e3f2dd1c81f2075171a547893391274'),
    ('tk',
     '6942ne4CqAM25cfC4EssHF4KPYKQwU5eZfkq2hxkeydl8lc3jlmZya3gKkwSeRGtnrmryRCGksGhdH6vvVel/MHsiFWkNBqc0k+5oLijPesWvVA='),
    ('type', 'default'),
    ('_', '1615095668350'),
)

while (True):
    resp = requests.get('https://passport.baidu.com/viewlog/getstyle', headers=headers, params=params,
                        cookies=cookies)

    respText = resp.content.decode("utf-8")
    print(respText)

    # 下载图片
    captchaUrl = urllib.parse.unquote(json.loads(respText)["data"]["ext"]["img"])
    resp = requests.get(captchaUrl, headers=headers, cookies=cookies)

    with open("D:\\PycharmProjects\\rotnet\\data\\baiduCaptcha\\{}.jpg".format(int(time.time())), "wb") as f:
        f.write(resp.content)

    time.sleep(1)
