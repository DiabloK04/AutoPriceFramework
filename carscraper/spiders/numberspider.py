import scrapy
from urllib.parse import urljoin
from itemloaders import ItemLoader
from carscraper.items import CarscraperItem


class numberSpider(scrapy.Spider):
    name = "numberspider"
    allowed_domains = ["autoscout24.nl"]
    base_url = "https://www.autoscout24.nl/lst/"

    mileageList = [
        ("0", "20000"), ("20000", "50000"), ("50000", "80000"), ("80000", "100000"), ("100000", "125000"),
        ("125000", "150000"), ("150000", "175000"), ("175000", "200000")
    ]
    years = list(range(1995, 2025,5))

    makeList = [
        'abarth', 'alfa-romeo', 'aston-martin', 'audi', 'bentley',
        'bmw', 'chevrolet', 'chrysler', 'citroen', 'dacia', 'daihatsu', 'dodge',
        'fiat', 'ford', 'honda', 'hyundai', 'iveco', 'jaguar', 'jeep', 'kia',
        'lancia', 'land-rover', 'lexus', 'maserati', 'mazda', 'mercedes-benz', 'mg', 'mini',
        'mitsubishi', 'nissan', 'opel', 'peugeot', 'polestar', 'porsche', 'renault',
        'rolls-royce', 'saab', 'seat', 'skoda', 'smart', 'subaru', 'suzuki', 'tesla',
        'toyota', 'volkswagen', 'volvo'
    ]

    def start_requests(self):
        for make in self.makeList:
            for mileage_from, mileage_to in self.mileageList:
                for year in self.years:
                    url = (f"{self.base_url}{make}?adage=14&atype=C&cy=NL&damaged_listing=exclude&desc=0"
                           f"&fregfrom={year}&fregto={year + 5}&kmfrom={mileage_from}&kmto={mileage_to}&offer=U&atype=C&page=1")
                    yield scrapy.Request(url, self.parse_pagination,
                                         meta={'make': make, 'year': year, 'mileage': (mileage_from, mileage_to)})

    def parse_pagination(self, response):
        nrcars_text = response.css("h1::text").get()
        if not nrcars_text:
            return  # No cars found for this filter

        nr, _ = nrcars_text.split(" ", maxsplit=1)
        try:
            nr = int(nr.replace(".", ""))  # Convert to integer, removing any thousand separators
        except ValueError:
            return  # Skip if parsing fails

        total_pages = (nr // 19) + 1



        #Each page contains up to 19 cars
        make = response.meta['make']
        year = response.meta['year']
        if total_pages > 20:
            print(make, year)
        mileage_from, mileage_to = response.meta['mileage']

        for page in range(1, total_pages + 1):
            url = (f"{self.base_url}{make}?adage=7&atype=C&cy=NL&damaged_listing=exclude&desc=0"
                   f"&fregfrom={year}&fregto={year + 5}&kmfrom={mileage_from}&kmto={mileage_to}&offer=U&atype=C&page={page}")
            yield scrapy.Request(url, self.parse_listings)

    def parse_listings(self, response):
        item_urls = response.css(
            "a.ListItem_title__ndA4s.ListItem_title_new_design__QIU2b.Link_link__Ajn7I::attr(href)").getall()
        for url in item_urls:
            yield scrapy.Request(urljoin(response.url, url), callback=self.parse_item)

    def parse_item(self, response):
        self.log(f"Visited {response.url}")
        loader = ItemLoader(item=CarscraperItem(), selector=response)
        loader.add_xpath("make",'//*[@class="StageTitle_boldClassifiedInfo__sQb0l StageTitle_textOverflow__KN9BA"]/text()')
        loader.add_xpath("model",'//*[@class="StageTitle_boldClassifiedInfo__sQb0l StageTitle_textOverflow__KN9BA"]/text()')
        loader.add_xpath("price_euro", '//*[contains(@class, "PriceInfo_price__XU0aF")]/text()')
        loader.add_xpath("year", '//*[@id="listing-history-section"]//*[text() = "Bouwjaar"]/following::dd[1]/text()')
        loader.add_xpath("transmission",'//*[@id="technical-details-section"]//*[text() = "Transmissie"]/following::dd[1]/text()')
        loader.add_xpath("power",'//*[@id="technical-details-section"]//*[text() = "Vermogen kW (PK)"]/following::dd[1]/text()')
        loader.add_xpath("weight",'//*[@id="technical-details-section"]//*[text() = "Leeggewicht"]/following::dd[1]/text()')
        loader.add_xpath("fuel",'//*[@id="environment-details-section"]//*[text() = "Brandstof"]/following::dd[1]/text()')
        loader.add_xpath("fuel",'//*[@id="environment-details-section"]//*[text() = "Andere energiebronnen"]/following::dd[1]/text()')
        loader.add_xpath("mileage",'//*[@id="listing-history-section"]//*[text() = "Kilometerstand"]/following::dd[1]/div/span/text()')
        loader.add_xpath("owners",'//*[@id="listing-history-section"]//*[text() = "Vorige eigenaren"]/following::dd[1]/text()')
        loader.add_xpath("history",'//*[@id="listing-history-section"]//*[text() = "Volledige onderhoudshistorie"]/following::dd[1]/text()')
        loader.add_xpath("color", '//*[@id="color-section"]//*[text() = "Kleur"]/following::dd[1]/text()')
        loader.add_xpath("seller", '//h2[@class="VendorData_titleLabel__2ZuoZ"]/following::span[1]/text()')
        loader.add_xpath("description", '//*[(@id = "sellerNotesSection")]//div/text()')
        loader.add_xpath("guarantee",'//*[(@id = "basic-details-section")]//*[text() = "Garantie"]/following::dd[1]/text()')
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




