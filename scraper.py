import time
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse, unquote

from model import PropertyItem


class ScraperLoggerAdapter(logging.LoggerAdapter):
    """Додає назву платформи перед логом."""

    def process(self, msg, kwargs):
        return f"[{self.extra['platform']}] {msg}", kwargs


class BaseScraper:
    platform = "Base"
    base_domain = "https://example.com"

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        }

        base_logger = logging.getLogger(self.__class__.__name__)
        self.logger = ScraperLoggerAdapter(base_logger, {"platform": self.platform})

    def fetch_html(self, url: str) -> Optional[BeautifulSoup]:
        soup = None
        try:
            time.sleep(0.25)
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            self.logger.error(f"Помилка завантаження {url}: {e}")
        return soup

    def div_to_text(self, soup, attribute: Tuple[str, str]) -> Optional[str]:
        text = None
        div = soup.find("div", {attribute[0]: attribute[1]})
        if div:
            text = div.get_text(separator="\n", strip=True)
        else:
            self.logger.warning(f"Блок {attribute[0]}={attribute[1]} не знайдено.")
        return text

    def _build_paginated_url(self, base_url: str, page: int) -> str:
        if page == 1:
            return base_url

        parsed_url = urlparse(base_url)
        query_params = parse_qsl(parsed_url.query, keep_blank_values=True)
        query_params = [param for param in query_params if param[0] != "page"]

        query_params.append(("page", str(page)))

        new_query = urlencode(query_params)
        new_url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                new_query,
                parsed_url.fragment,
            )
        )

        return unquote(new_url)

    def scrap_category(
        self, category: str, url: str, limit: int = 5, max_pages: int = 3
    ) -> List[PropertyItem]:
        self.logger.info(f"Запуск збору категорії: {category} | Мета: {limit} карток")

        items = []

        for page in range(1, max_pages + 1):
            if len(items) >= limit:
                break

            page_url = self._build_paginated_url(url, page)
            self.logger.info(f"Завантаження сторінки {page}: {page_url}")

            category_soup = self.fetch_html(page_url)
            if not category_soup:
                self.logger.warning(f"Не вдалося завантажити сторінку списку {page}.")
                break  # Зупиняємо пагінацію, якщо сторінка не відповідає

            cards = self.extract_cards(category_soup)
            if not cards:
                self.logger.info(f"На сторінці {page} порожньо. Завершення пагінації.")
                break

            self.logger.info(f"Знайдено карток на сторінці {page}: {len(cards)}")

            for card in cards:
                if len(items) >= limit:
                    break

                href = self.extract_href(card)
                if not href:
                    continue

                ad_url = f"{self.base_domain}{href}" if href.startswith("/") else href
                self.logger.info(
                    f"Завантаження оголошення [{len(items) + 1}/{limit}]: {ad_url}"
                )

                item = self.scrap_ad(category, ad_url)
                if item:
                    items.append(item)

        return items

    def scrap_ad(self, category: str, url: str) -> Optional[PropertyItem]:
        item = None
        try:
            soup = self.fetch_html(url)
            if soup:
                desc = self.parse_desc(soup)
                if desc:
                    item = PropertyItem(self.platform, category, url, desc)
        except Exception as e:
            self.logger.error(f"Помилка під час обробки посилання {url}: {e}")

        return item

    def extract_cards(self, soup: BeautifulSoup) -> list:
        raise NotImplementedError("Реалізуй у дочірньому класі.")

    def extract_href(self, card) -> Optional[str]:
        raise NotImplementedError("Реалізуй у дочірньому класі.")

    def parse_desc(self, soup: BeautifulSoup) -> Optional[str]:
        raise NotImplementedError("Реалізуй у дочірньому класі.")


class OLXScraper(BaseScraper):
    platform = "OLX"
    base_domain = "https://www.olx.ua"

    def extract_cards(self, soup: BeautifulSoup) -> list:
        return soup.find_all("div", {"data-cy": "l-card"})

    def extract_href(self, card) -> Optional[str]:
        a_tag = card.find("a", href=True)
        return a_tag["href"] if a_tag else None

    def parse_desc(self, soup: BeautifulSoup) -> Optional[str]:
        return self.div_to_text(soup, ("data-testid", "ad_description"))


class DIMRIAScrapper(BaseScraper):
    platform = "DIMRIA"
    base_domain = "https://dom.ria.com"

    def extract_cards(self, soup: BeautifulSoup) -> list:
        return soup.find_all("section", class_="realty-item")

    def extract_href(self, card) -> Optional[str]:
        link_tag = card.find("a", class_="realty-link")
        return link_tag["href"] if link_tag and "href" in link_tag.attrs else None

    def parse_desc(self, soup: BeautifulSoup) -> Optional[str]:
        return self.div_to_text(soup, ("id", "descriptionBlock"))
