import dataclasses
import json
import logging
import os
import pandas as pd

from scraper import OLXScraper, DIMRIAScrapper


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Визначення джерел
    try:
        sources = pd.read_csv("sources.csv")
    except FileNotFoundError:
        logging.error("Файл sources.csv не знайдено.")
        return

    # 2. Ініціалізація компонентів
    olx_scraper = OLXScraper()
    dimria_scraper = DIMRIAScrapper()

    all_properties = []

    # 3. Збір даних
    logging.info("Збір даних...")
    for source in sources.to_dict("records"):
        platform = source["platform"]
        category = source["category"]
        url = source["url"]

        if platform == "OLX":
            scraper = olx_scraper
        elif platform == "DIM.RIA":
            scraper = dimria_scraper
        else:
            continue

        scraped_items = scraper.scrap_category(category, url, limit=50)

        if not scraped_items:
            logging.warning(f"Немає даних для збереження з {platform}/{category}")
            continue

        save_dir = os.path.join("data", category, platform)
        os.makedirs(save_dir, exist_ok=True)

        # Збереження у локальний файл properties.jsonl
        local_filepath = os.path.join(save_dir, "properties.jsonl")
        with open(local_filepath, "w", encoding="utf-8") as f:
            for item in scraped_items:
                json_line = json.dumps(dataclasses.asdict(item), ensure_ascii=False)
                f.write(json_line + "\n")

        all_properties.extend(scraped_items)

        logging.info(
            f"Успішно збережено {len(scraped_items)} PropertyItems у {save_dir}\n"
        )

    if not all_properties:
        logging.error("Жодного об'єкта не було успішно розпарсено.")
        return

    # 4. Збереження у файл загальної бази
    global_filepath = "properties.jsonl"
    with open(global_filepath, "w", encoding="utf-8") as f:
        for item in all_properties:
            json_line = json.dumps(dataclasses.asdict(item), ensure_ascii=False)
            f.write(json_line + "\n")

    logging.info(
        f"Загальну базу з {len(all_properties)} об'єктів успішно збережено у {
            global_filepath
        }"
    )


if __name__ == "__main__":
    main()
