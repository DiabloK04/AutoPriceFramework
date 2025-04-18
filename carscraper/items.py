# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
from os import remove
import scrapy
from scrapy import Field
from itemloaders.processors import Join, MapCompose, TakeFirst



def get_price(txt: str):
    if not txt:  # Handle None or empty values
        return None

        # Remove currency symbols and unwanted characters
    txt = txt.replace("€", "").replace(",-", "").strip()

    # Convert thousands separator (.) to nothing and decimal separator (,) to a dot
    txt = txt.replace(".", "").replace(",", ".")

    try:
        return float(txt)  # Convert to float
    except ValueError:
        return None  # Return None if conversion fails


def get_make_model(txt: str, getMake: bool):
    if not txt:  # Handle None or empty values
        return None

    if ' ' in txt:  # Check if there's a space to split
        [make, model] = txt.split(" ", maxsplit=1)
        if getMake: return make
        return model
    else:
        if getMake: return txt
        return None



def get_nr(txt: str):
    # Remove all periods (.)
    input_string = txt.replace('.', '')
    # Split the string based on spaces and take the first part
    processed_input = input_string.split()[0]
    return processed_input

def join_text(text_list):
    # Join all the text elements into one continuous string
    return ' '.join(text_list)

# Example usage

class CarscraperItem(scrapy.Item):
    price_euro = Field(input_processor=MapCompose(get_price), output_processor = TakeFirst())
    make = Field(input_processor = MapCompose(lambda txt: get_make_model(txt, True)), output_processor = TakeFirst())
    model = Field(input_processor = MapCompose(lambda txt: get_make_model(txt, False)), output_processor = TakeFirst())
    mileage = Field(input_processor = MapCompose(get_nr), output_processor = TakeFirst())
    power = Field(input_processor = MapCompose(get_nr), output_processor = TakeFirst())
    description = Field(input_processor = join_text, output_processor = TakeFirst())
    year = Field(output_processor = TakeFirst())
    transmission = Field(output_processor = TakeFirst())
    fuel = Field(output_processor = TakeFirst())
    owners = Field(input_processor = MapCompose(get_nr), output_processor = TakeFirst())
    history = Field(output_processor = TakeFirst())
    color = Field(output_processor = TakeFirst())
    seller = Field(output_processor = TakeFirst())
    weight = Field(input_processor = MapCompose(get_nr), output_processor = TakeFirst())
    guarantee = Field(input_processor = MapCompose(get_nr), output_processor = TakeFirst())
    apk = Field(output_processor = TakeFirst())



