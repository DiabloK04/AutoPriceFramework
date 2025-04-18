import scrapy
from urllib.parse import urljoin
from itemloaders import ItemLoader
from carscraper.items import CarscraperItem
from itemloaders.processors import Join, MapCompose, TakeFirst



class CarspiderSpider(scrapy.Spider):
    name = "carspider"
    allowed_domains = ["autoscout24.nl"]
    start_urls = ["https://www.autoscout24.nl"]

    def parse(self, response):

        page_count = 20

        for idx in range(1, page_count + 1):
             yield scrapy.Request(
                'https://www.autoscout24.nl/lst?cy=NL&damaged_listing=exclude&desc=0&offer=U&powertype=kw&search_id=z6ctle9fm0&sort=standard&source=homepage_search-mask&ustate=N%2CU'
                + '&atype=C&page='
                + str(idx)
             )

        item_selector = response.css("a.ListItem_title__ndA4s.ListItem_title_new_design__QIU2b.Link_link__Ajn7I::attr(href)").getall()
        for url in item_selector:
            yield scrapy.Request(urljoin(response.url, url), callback=self.parse_item)





    def parse_item(self, response):
        self.log("Visited %s" % response.url)
        loader = ItemLoader(item=CarscraperItem(), selector=response)

        loader.add_xpath("make", '//*[@class="StageTitle_boldClassifiedInfo__sQb0l StageTitle_textOverflow__KN9BA"]/text()')
        loader.add_xpath("model",'//*[@class="StageTitle_boldClassifiedInfo__sQb0l StageTitle_textOverflow__KN9BA"]/text()')
        loader.add_xpath("price_euro", '//*[contains(@class, "PriceInfo_price__XU0aF")]/text()')
        loader.add_xpath("year", '//*[@id="listing-history-section"]//*[text() = "Bouwjaar"]/following::dd[1]/text()')
        loader.add_xpath("transmission",'//*[@id="technical-details-section"]//*[text() = "Transmissie"]/following::dd[1]/text()')
        loader.add_xpath("power",'//*[@id="technical-details-section"]//*[text() = "Vermogen kW (PK)"]/following::dd[1]/text()')
        loader.add_xpath("fuel",'//*[@id="environment-details-section"]//*[text() = "Brandstof"]/following::dd[1]/text()')
        loader.add_xpath("fuel",'//*[@id="environment-details-section"]//*[text() = "Andere energiebronnen"]/following::dd[1]/text()')
        loader.add_xpath("mileage",'//*[@id="listing-history-section"]//*[text() = "Kilometerstand"]/following::dd[1]/div/span/text()')
        loader.add_xpath("owners",'//*[@id="listing-history-section"]//*[text() = "Vorige eigenaren"]/following::dd[1]/text()')
        loader.add_xpath("history", '//*[@id="listing-history-section"]//*[text() = "Volledige onderhoudshistorie"]/following::dd[1]/text()')
        loader.add_xpath("color", '//*[@id="color-section"]//*[text() = "Kleur"]/following::dd[1]/text()')
        loader.add_xpath("seller", '//h2[@class="VendorData_titleLabel__2ZuoZ"]/following::span[1]/text()')
        loader.add_xpath("description", '//*[(@id = "sellerNotesSection")]//div/text()')
        loader.add_xpath("guarantee", '//*[(@id = "basic-details-section")]//*[text() = "Garantie"]/following::dd[1]/text()')
        loader.add_xpath("apk", '//*[@id="listing-history-section"]//*[text() = "APK"]/following::dd[1]/text()')

        detail_url = response.css("a.CarReports_button__AkcyE::attr(href)").get()
        if detail_url:
            # Clone loader to prevent conflicts if we yield it now
            base_item = loader.load_item()
            yield base_item  # yield now so we always get the base version

            # Now enrich via detail page
            yield response.follow(
                detail_url,
                callback=self.parse_detail,
                meta={'loader': loader}
            )
        else:
            yield loader.load_item()

    def parse_detail(self, response):
        loader = response.meta.get('loader')
        if not loader:
            self.logger.warning("Loader not found in detail page")
            return

        loader.selector = response
        loader.add_xpath('owners', '//h5[text() = "Aantal eigenaren"]/following::span[1]/text()')

        yield loader.load_item()