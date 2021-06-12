from selenium import webdriver
from bs4 import BeautifulSoup

import csv
import time
import pandas as pd


t_url = "https://www.kbchachacha.com/public/search/main.kbc#!?countryOrder=1&page=" #URL

for i in range(1,501): # 받고 싶은 페이지 설정
    j = str(i)
    url = t_url + j
    print("\n홈페이지:" +url + "에서 크롤링중")

    driver = webdriver.Chrome('chromedriver.exe')
    driver.get(url)



    ##한페이지에서 href주소 가져오기
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    notices = soup.select('div.generalRegist > div.list-in > div.area > div.con > div.item > a')

    for i in range (20):
        tt_url = "https://www.kbchachacha.com/"
        url2 = tt_url + notices[i]["href"]
        driver.get(url2)


        #한페이지의 데이터를 엑셀로 저장

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        #차량 기본정보
        car_info=soup.select('#content > div:nth-child(15) > div.common-sub-content.common-container.fix-content > div.cmm-cardt-area.adj1740.adj1670.adj1500.adj1441 > div.car-detail-info > div > div.detail-info01 > table > tbody')

        #차량이름
        car_title=soup.select('#content > div:nth-child(15) > div.common-sub-content.common-container.fix-content > div.cmm-cardt-area.adj1740.adj1670.adj1500.adj1441 > div.car-dt-info > div.car-buy-info > div.car-buy-price > strong')

        #차량가격
        car_price=soup.select('#content > div:nth-child(15) > div.common-sub-content.common-container.fix-content > div.cmm-cardt-area.adj1740.adj1670.adj1500.adj1441 > div.car-dt-info > div.car-buy-info > div.car-buy-price > div > dl > dd > strong')

        if not car_price or not car_title or not car_info:
            continue
        #차량이름, 가격, 제조사
        title=car_title[0].text
        series=title.split(' ',1)[1]
        
        price=car_price[0].text
        madein=title.split(' ')[0].split(')')[1]
        
        #차량정보 태그
        info_th=car_info[0].find_all('th')
        info_td=car_info[0].find_all('td')

        info_th_text=[]
        info_td_text=[]

        #차량정보배열
        for i in range (len(info_th)):
            info_th_text.append(info_th[i].text.replace('\t',"").replace('\n',"").replace('Km',"").replace('km',"").replace('cc',"").strip())
            info_td_text.append(info_td[i].text.replace('\t',"").replace('\n',"").replace('Km',"").replace('km',"").replace('cc',"").strip())

        info_td_text.append(series)
        info_td_text.append(madein)
        info_td_text.append(price)
        
        #차량정보 출력
        print(info_td_text)
        #정보 csv에 저장하기
        f = open('파일제목.csv','a',encoding='utf-8-sig', newline='')
        wr = csv.writer(f)
        wr.writerow(info_td_text) 
        f.close()
driver.close()












    
