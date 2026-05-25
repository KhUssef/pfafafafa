"""
Booking.com Attractions Scraper using Selenium with Edge
Scrapes attractions data from Booking.com using stable selectors only.

Strategy:
- ONLY use data-testid, aria-*, role, id attributes — never css-xxxxxxx classes
- Extract review info from the screen-reader aria-label span (always present, structured)
- Extract price from aria-hidden price block text
- Fall back to structural/positional XPath where no stable attribute exists
"""

import json
import time
import random
import re
import argparse
import csv
from datetime import datetime, timedelta
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stable selectors — anchored to data-testid / aria / role / id only
# ---------------------------------------------------------------------------
SEL = {
    # Results page wrapper
    "results_container": "#attr-search-results-page-main-content",

    # Individual attraction card
    "card": "[data-testid='card']",

    # Title link inside the card
    "title_link": "[data-testid='card-title'] a",

    # Review block — the screen-reader span always holds
    # "User reviews, 3.9 out of 5 stars from 37 reviews"
    "review_sr_text": "[data-testid='review-score'] .bc946a29db",

    # Price block — aria-hidden div contains "From  €  35" as structured spans
    "price_block": "[data-testid='price'] [aria-hidden='true']",

    # Availability text — first sibling div after the price block
    # We use XPath for this because there's no stable attribute
    # (see _extract_availability below)

    # Free cancellation badge — look for this specific checkmark SVG wrapper text
    "free_cancel": "span.cff4a33cd8",

    # Duration chip — contains "Duration:" text
    # Located inside .css-* but we can target the text div next to the clock icon
    # We use XPath: //div[starts-with(text(),'Duration')]
}

# XPath patterns used when CSS isn't sufficient
XPATH = {
    "search_input":     "//input[contains(@placeholder, 'Where are you going')]",
    "suggestion_first": (
        "//div[@role='listbox']//*[@role='option'][1]"
        " | //div[@role='listbox']//button[1]"
        " | //div[@role='listbox']//a[1]"
    ),
    "date_button":      "//button[contains(@aria-label, 'Select dates')]",
    "date_dialog":      "//div[@role='dialog'][@aria-label='Select dates']",
    "month_header":     "//h3[starts-with(@id,'bui-calendar-month-')]",
    "next_month":       "//button[@aria-label='Next month']",
    "prev_month":       "//button[@aria-label='Previous month']",
    "search_button":    (
        "//button[normalize-space(text())='Search']"
        " | //button[@type='submit' and not(@aria-label)]"
    ),
    "duration_text":    ".//div[starts-with(normalize-space(text()),'Duration')]",
    "availability_text": (
        # Sibling div after [data-testid='price'] with text like "Available from …"
        "./following-sibling::div[contains(normalize-space(.),'Available')]"
        " | ./following-sibling::div[contains(normalize-space(.),'available')]"
    ),
    # Popup close buttons
    "popups": [
        "//button[@aria-label='Dismiss sign in information.']",
        "//button[@aria-label='Dismiss sign-in info.']",
        "//button[@id='onetrust-accept-btn-handler']",
        "//button[@aria-label='Close']",
        "//div[@role='dialog']//button[@aria-label='Close']",
        "//button[@aria-label='Dismiss']",
        "//button[@data-testid='header-signin-link']",   # sometimes overlaps
    ],
}


def _parse_review_sr(text: str) -> dict:
    """
    Parse the screen-reader review string.

    Input:  "User reviews, 3.9 out of 5 stars from 37 reviews"
    Output: {"review_score": "3.9", "review_label": "", "review_count": "37"}

    The label ("Good", "Wonderful", etc.) is not in the SR text, so we also
    try to read it from the adjacent visible spans if provided separately.
    """
    result = {"review_score": "N/A", "review_label": "N/A", "review_count": "N/A"}
    if not text:
        return result

    score_m = re.search(r"([\d.]+)\s+out of", text)
    count_m = re.search(r"from\s+(\d+)\s+review", text)

    if score_m:
        result["review_score"] = score_m.group(1)
    if count_m:
        result["review_count"] = count_m.group(1)

    return result


def _parse_price_block(text: str) -> str:
    """
    Parse the aria-hidden price block text.

    The block contains text nodes from multiple child spans, so after
    get_attribute('innerText') or .text we might get something like:
        "From\n€\xa035"   or   "From\n€ 35"
    Normalise to "From €35".
    """
    if not text or text.strip() == "":
        return "N/A"
    # Collapse whitespace and non-breaking spaces
    cleaned = re.sub(r"[\s\u00a0]+", " ", text).strip()
    return cleaned


class BookingAttractionsScraper:
    """Scraper for Booking.com attractions — stable-selector edition."""

    BASE_URL = "https://www.booking.com/attractions/"

    def __init__(self, headless: bool = False):
        self.driver = None
        self.headless = headless
        self._setup_driver()

    # ------------------------------------------------------------------
    # Driver setup
    # ------------------------------------------------------------------

    def _setup_driver(self):
        options = webdriver.EdgeOptions()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        # Disable image loading for speed
        options.add_experimental_option("prefs", {
            "profile.default_content_settings.images": 2,
            "profile.managed_default_content_settings.images": 2,
        })
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
        )
        self.driver = webdriver.Edge(options=options)
        self.driver.set_window_size(1920, 1080)
        logger.info("Edge driver initialised")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sleep(self, lo: float = 0.5, hi: float = 2.0):
        time.sleep(random.uniform(lo, hi))

    def _type(self, element, text: str, lo: float = 0.05, hi: float = 0.14):
        """Human-like keystroke typing."""
        for ch in text:
            element.send_keys(ch)
            time.sleep(random.uniform(lo, hi))

    def _js_click(self, element):
        self.driver.execute_script("arguments[0].click();", element)

    def _wait(self, timeout: int = 10) -> WebDriverWait:
        return WebDriverWait(self.driver, timeout)

    # ------------------------------------------------------------------
    # Cookies
    # ------------------------------------------------------------------

    def load_cookies(self, cookies_file: str) -> bool:
        try:
            with open(cookies_file) as f:
                cookies = json.load(f)
            self.driver.get(self.BASE_URL)
            self._sleep(0.5, 1)
            for cookie in cookies:
                c = {k: v for k, v in cookie.items()
                     if k not in ("expirationDate", "sameSite", "storeId")}
                if "expirationDate" in cookie and "expiry" not in c:
                    c["expiry"] = int(cookie["expirationDate"])
                try:
                    self.driver.add_cookie(c)
                except Exception as exc:
                    logger.debug("Cookie %s skipped: %s", cookie.get("name"), exc)
            logger.info("Loaded %d cookies", len(cookies))
            return True
        except FileNotFoundError:
            logger.warning("Cookies file not found: %s", cookies_file)
            return False
        except Exception as exc:
            logger.error("Error loading cookies: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Popup dismissal
    # ------------------------------------------------------------------

    def _dismiss_popups(self, timeout: int = 5):
        """Try all known popup-close XPaths."""
        for xpath in XPATH["popups"]:
            try:
                buttons = self.driver.find_elements(By.XPATH, xpath)
                for btn in buttons[:2]:
                    try:
                        if btn.is_displayed():
                            self._js_click(btn)
                            logger.debug("Dismissed popup: %s", xpath)
                            self._sleep(0.4, 0.8)
                    except StaleElementReferenceException:
                        pass
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Date picker
    # ------------------------------------------------------------------

    def _select_date(self, target_date: str) -> bool:
        """
        Select a date in the Booking.com calendar dialog.

        Args:
            target_date: "YYYY-MM-DD"
        """
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD.")
            return False

        target_month_str = dt.strftime("%B %Y")   # e.g. "June 2026"
        date_css = f"span[data-date='{target_date}'][role='checkbox']"

        try:
            # Open the calendar
            btn = self._wait(10).until(
                EC.presence_of_element_located((By.XPATH, XPATH["date_button"]))
            )
            if not self.driver.find_elements(By.XPATH, XPATH["date_dialog"]):
                try:
                    btn.click()
                except Exception:
                    self._js_click(btn)
            self._sleep(0.2, 0.4)

            # Wait for dialog
            self._wait(5).until(
                EC.presence_of_element_located((By.XPATH, XPATH["date_dialog"]))
            )

            # Navigate to the correct month (max 24 steps)
            for _ in range(24):
                header = self.driver.find_element(By.XPATH, XPATH["month_header"])
                if header.text == target_month_str:
                    break
                current = datetime.strptime(header.text, "%B %Y")
                nav_xpath = XPATH["next_month"] if dt > current else XPATH["prev_month"]
                self.driver.find_element(By.XPATH, nav_xpath).click()
                self._sleep(0.3, 0.6)
            else:
                logger.warning("Could not navigate to %s", target_month_str)
                return False

            # Click the day
            day_el = self._wait(5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, date_css))
            )
            day_el.click()
            logger.info("Selected date: %s", target_date)
            self._sleep(0.8, 1.5)
            return True

        except Exception as exc:
            logger.error("Date selection failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, city: str, country: str, date: str | None = None) -> bool:
        """
        Navigate to Booking.com attractions, enter city/country, pick a date,
        and submit the search.

        Args:
            city:    e.g. "Nice"
            country: e.g. "France"
            date:    "YYYY-MM-DD" (default: tomorrow)
        """
        if not date:
            date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        self.driver.get(self.BASE_URL)
        self._sleep(0.3, 0.7)
        self._dismiss_popups()

        try:
            # Fill the search box
            search_input = self._wait(10).until(
                EC.presence_of_element_located((By.XPATH, XPATH["search_input"]))
            )
            search_input.clear()
            self._type(search_input, f"{city}, {country}")
            self._sleep(0.1, 0.3)

            # Select first autocomplete suggestion
            for attempt in range(3):
                try:
                    suggestion = self._wait(5).until(
                        EC.presence_of_element_located(
                            (By.XPATH, XPATH["suggestion_first"])
                        )
                    )
                    self._sleep(0.05, 0.15)
                    suggestion.click()
                    logger.info("Selected autocomplete suggestion")
                    break
                except StaleElementReferenceException:
                    if attempt == 2:
                        logger.warning("Autocomplete suggestion went stale; continuing")
                except TimeoutException:
                    logger.warning("No autocomplete suggestions appeared")
                    break
            self._sleep(0.1, 0.3)

            # Pick the date
            self._select_date(date)
            self._sleep(0.1, 0.3)

            # Submit
            try:
                search_btn = self._wait(8).until(
                    EC.element_to_be_clickable((By.XPATH, XPATH["search_button"]))
                )
                self._sleep(0.05, 0.15)
                self._js_click(search_btn)
                logger.info("Clicked Search button")
            except TimeoutException:
                logger.warning("Search button not found — trying any visible button")
                for btn in self.driver.find_elements(By.TAG_NAME, "button"):
                    if btn.is_displayed():
                        self._js_click(btn)
                        break

            # Wait for results
            self._sleep(4, 7)
            self._dismiss_popups()
            return True

        except Exception as exc:
            logger.error("Search failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def _extract_text(self, parent, css: str, default: str = "N/A") -> str:
        """Safe single-element text extraction by CSS selector."""
        try:
            return parent.find_element(By.CSS_SELECTOR, css).text.strip() or default
        except (NoSuchElementException, StaleElementReferenceException):
            return default

    def _extract_text_xpath(self, parent, xpath: str, default: str = "N/A") -> str:
        """Safe single-element text extraction by XPath."""
        try:
            return parent.find_element(By.XPATH, xpath).text.strip() or default
        except (NoSuchElementException, StaleElementReferenceException):
            return default

    def _extract_location_and_description(self, card) -> tuple[str, str]:
        """
        Location and description have no stable attributes — extract them
        positionally relative to the card-title heading.

        The DOM order inside the card text column is reliably:
          1. [data-testid='card-title']  ← heading
          2. div with location text      ← first div sibling after title's parent
          3. div with description text   ← next sibling
        """
        try:
            # The title is wrapped in a div > h3, so we go up two levels then
            # grab the next sibling divs using XPath from the card root.
            title_h3 = card.find_element(By.CSS_SELECTOR, "[data-testid='card-title']")
            # Navigate: h3 -> parent div -> parent div, then find next siblings
            parent_div = self.driver.execute_script(
                "return arguments[0].parentElement.parentElement;", title_h3
            )
            siblings = parent_div.find_elements(By.XPATH, "following-sibling::div")
            location = siblings[0].text.strip() if len(siblings) > 0 else "N/A"
            description = siblings[1].text.strip() if len(siblings) > 1 else "N/A"
            return location or "N/A", description or "N/A"
        except Exception:
            return "N/A", "N/A"

    def _extract_review(self, card) -> dict:
        """
        Extract review data from the stable screen-reader span inside
        [data-testid='review-score'].

        The SR span always contains structured text like:
          "User reviews, 3.9 out of 5 stars from 37 reviews"

        We also try to grab the visible label ("Good", "Wonderful", etc.)
        from the aria-hidden sibling spans.
        """
        result = {"review_score": "N/A", "review_label": "N/A", "review_count": "N/A"}
        try:
            review_block = card.find_element(
                By.CSS_SELECTOR, "[data-testid='review-score']"
            )
        except NoSuchElementException:
            return result

        # Screen-reader text → score + count
        sr_text = self._extract_text(review_block, ".bc946a29db")
        parsed = _parse_review_sr(sr_text)
        result.update(parsed)

        # Visible label — look for a span that contains text like "Good" / "Wonderful"
        # It is inside aria-hidden="true" spans; we grab all text and filter
        try:
            visible_spans = review_block.find_elements(
                By.XPATH,
                ".//span[@aria-hidden='true']//span[@aria-hidden='true']"
            )
            for span in visible_spans:
                t = span.text.strip()
                # The label span contains e.g. " · Good" — strip the dot
                clean = re.sub(r"^[\s·•\-]+", "", t).strip()
                if clean and not re.match(r"^[\d.]+$", clean):
                    result["review_label"] = clean
                    break
        except Exception:
            pass

        return result

    def _extract_price(self, card) -> str:
        """
        Extract price from [data-testid='price'] [aria-hidden='true'].

        The aria-hidden div holds the visual price (e.g. "From  €  35").
        We normalise whitespace and non-breaking spaces.
        """
        try:
            price_el = card.find_element(By.CSS_SELECTOR, SEL["price_block"])
            raw = price_el.get_attribute("innerText") or price_el.text
            return _parse_price_block(raw)
        except NoSuchElementException:
            return "N/A"

    def _extract_availability(self, card) -> str:
        """
        Extract availability text (e.g. "Available from 22 Jun").
        It lives in a sibling div after [data-testid='price'] with no stable id.
        We use XPath from the price element.
        """
        try:
            price_el = card.find_element(By.CSS_SELECTOR, "[data-testid='price']")
            avail = price_el.find_element(By.XPATH, XPATH["availability_text"])
            return avail.text.strip() or "N/A"
        except NoSuchElementException:
            return "N/A"

    def _extract_duration(self, card) -> str:
        """
        Extract duration text (e.g. "Duration: 4 hours - 5 hours").
        We search for any div whose text starts with "Duration".
        """
        return self._extract_text_xpath(card, XPATH["duration_text"])

    def scrape(self) -> list[dict]:
        """
        Scrape all attraction cards from the current results page.
        Returns a list of attraction dicts.
        """
        attractions = []

        try:
            container = self._wait(15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SEL["results_container"])
                )
            )
        except TimeoutException:
            logger.error("Results container not found — page may not have loaded")
            return attractions

        cards = container.find_elements(By.CSS_SELECTOR, SEL["card"])
        logger.info("Found %d attraction cards", len(cards))

        for idx, card in enumerate(cards):
            try:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'});", card
                )
                self._sleep(0.2, 0.5)

                attraction: dict = {}

                # Title + URL
                try:
                    title_el = card.find_element(By.CSS_SELECTOR, SEL["title_link"])
                    attraction["title"] = title_el.text.strip() or "N/A"
                    attraction["url"] = title_el.get_attribute("href") or "N/A"
                except NoSuchElementException:
                    attraction["title"] = "N/A"
                    attraction["url"] = "N/A"

                # Location + Description (positional fallback)
                loc, desc = self._extract_location_and_description(card)
                attraction["location"] = loc
                attraction["description"] = desc

                # Duration
                attraction["duration"] = self._extract_duration(card)

                # Reviews
                review_data = self._extract_review(card)
                attraction.update(review_data)

                # Price
                attraction["price"] = self._extract_price(card)

                # Availability
                attraction["availability"] = self._extract_availability(card)

                # Free cancellation flag
                try:
                    cancel_el = card.find_element(By.CSS_SELECTOR, SEL["free_cancel"])
                    attraction["free_cancellation"] = bool(cancel_el.text.strip())
                except NoSuchElementException:
                    attraction["free_cancellation"] = False

                attractions.append(attraction)
                logger.info(
                    "Card %d: %s | %s | %s",
                    idx + 1,
                    attraction["title"],
                    attraction["price"],
                    attraction["review_score"],
                )

            except StaleElementReferenceException:
                logger.warning("Stale element at index %d — skipping", idx)
            except Exception as exc:
                logger.error("Error on card %d: %s", idx, exc)

        return attractions

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def save_csv(attractions: list[dict], filename: str):
        if not attractions:
            logger.warning("No attractions to save")
            return
        keys = list(attractions[0].keys())
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(attractions)
        logger.info("Saved %d rows to %s", len(attractions), filename)

    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Booking.com attractions (stable-selector edition)"
    )
    parser.add_argument("--city",     required=True, help="City name, e.g. Nice")
    parser.add_argument("--country",  required=True, help="Country name, e.g. France")
    parser.add_argument("--date",     default=None,
                        help="Date YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--output",   default=None,
                        help="Output CSV filename")
    parser.add_argument("--cookies",  default="booking_cookies.json",
                        help="Path to cookies JSON file")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless")
    args = parser.parse_args()

    # Validate date
    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error("--date must be YYYY-MM-DD")
            return

    output = args.output or f"attractions_{args.city}_{args.country}.csv"
    scraper = BookingAttractionsScraper(headless=args.headless)

    try:
        scraper.load_cookies(args.cookies)

        logger.info("Searching: %s, %s on %s", args.city, args.country,
                    args.date or "tomorrow")
        if scraper.search(args.city, args.country, args.date):
            attractions = scraper.scrape()
            scraper.save_csv(attractions, output)
            logger.info("Done — %d attractions saved to %s",
                        len(attractions), output)
        else:
            logger.error("Search step failed")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
    finally:
        scraper.close()


if __name__ == "__main__":
    main()