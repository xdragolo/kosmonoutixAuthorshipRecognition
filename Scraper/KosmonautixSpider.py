import scrapy
from scrapy.crawler import CrawlerProcess
import re
from scrapy import cmdline
from scrapy.utils.project import get_project_settings


class KosmonautixSpider(scrapy.Spider):
    name = 'kosmonautix'

    def start_requests(self):
        baseUrl = 'https://kosmonautix.cz/'
        urls = [baseUrl]
        for n in range(2, 543):
            urls.append(baseUrl + 'page/' + str(n) + '/')
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        urls = response.xpath('//div[@id="content"]/div/h2/a/@href').getall()
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parseArticle)

    def parseArticle(self, response):
        # title
        title = response.xpath('//h2[@class="title"]/text()').get()
        # author
        author = response.xpath('//a[@class="author url fn"]/text()').get()
        # date
        date = response.xpath('//div[@class="postdate"]/text()').get().strip()
        # article
        divEntry = response.xpath('//div[@class="entry"]')
        content = ''
        for p in divEntry.xpath("./p"):
            text = p.xpath("string(.)").get()
            if not re.match('Přeloženo z:|Zdroje obrázků:|Zdroje informací:', text):
                text = text.replace('\xa0', ' ')
                content += text + '\n'

        yield {
            'title': title,
            'author': author,
            'date': date,
            'content': content
        }


settings = get_project_settings()
settings.update({
    'FEED_FORMAT': 'json',
    'FEED_URI': 'kosmonautix.json'
})

process = CrawlerProcess(settings)
process.crawl(KosmonautixSpider)
process.start()
