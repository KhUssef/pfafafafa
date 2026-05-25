"""
Booking.com Hotels Scraper using Selenium with Edge
Scrapes hotel data from Booking.com home page.
"""

import argparse
import csv
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BookingHotelsScraper:
    """Scraper for Booking.com hotels from homepage."""

    BASE_URL = "https://www.booking.com/"

    DATA_SELECTORS = {
        "card": "[data-testid='property-card']",
        "title": "[data-testid='title-link'], [data-testid='title']",
        "location": "[data-testid='address']",
        "price": "[data-testid='price-and-discounted-price'], [data-testid='price']",
        "review_score": "[data-testid='review-score']",
        "review_count": "[data-testid='review-score'] + div, [aria-label*='scored']",
        "link": "a[data-testid='title-link']",
    }

    def __init__(self, headless=False):
        self.driver = None
        self.headless = headless
        self._setup_driver()

    def _setup_driver(self):
        """Setup Edge driver with basic optimizations."""
        options = webdriver.EdgeOptions()

        if self.headless:
            options.add_argument("--headless")

        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-extensions")

        prefs = {
            "profile.default_content_settings.images": 2,
            "profile.managed_default_content_settings.images": 2,
        }
        options.add_experimental_option("prefs", prefs)

        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
        )
        options.add_argument(f"user-agent={user_agent}")

        self.driver = webdriver.Edge(options=options)
        self.driver.set_window_size(1920, 1080)
        logger.info("Edge driver initialized")

    def _load_cookies(self, cookies_file):
        """Load cookies from JSON file."""
        try:
            with open(cookies_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)

            self.driver.get(self.BASE_URL)
            time.sleep(2)

            for cookie in cookies:
                try:
                    cookie_copy = cookie.copy()
                    for key in ["expirationDate", "sameSite", "storeId"]:
                        cookie_copy.pop(key, None)

                    if "expiry" not in cookie_copy and "expirationDate" in cookie:
                        cookie_copy["expiry"] = int(cookie["expirationDate"])

                    self.driver.add_cookie(cookie_copy)
                except Exception as e:
                    logger.warning("Could not add cookie %s: %s", cookie.get("name"), e)

            logger.info("Loaded %s cookies", len(cookies))
            return True
        except FileNotFoundError:
            logger.error("Cookies file not found: %s", cookies_file)
            return False
        except Exception as e:
            logger.error("Error loading cookies: %s", e)
            return False

    def _human_sleep(self, min_sleep=0.5, max_sleep=2.0):
        """Sleep random time to mimic human behavior."""
        time.sleep(random.uniform(min_sleep, max_sleep))

    def _dismiss_popups(self, wait_time=5):
        """Dismiss common popups and cookie banners."""
        _ = WebDriverWait(self.driver, wait_time)

        popup_selectors_xpath = [
            "//button[@aria-label='Dismiss sign in information.']",
            "//button[@aria-label='Dismiss sign-in info.']",
            "//button[@id='onetrust-accept-btn-handler']",
            "//button[@aria-label='Close']",
            "//button[@data-modal-close='true']",
            "//div[@role='dialog']//button[@aria-label='Close']",
            "//button[@aria-label='Dismiss']",
            "//button[contains(@class, 'close')]",
            "//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept all')]",
        ]

        for selector in popup_selectors_xpath:
            try:
                buttons = self.driver.find_elements(By.XPATH, selector)
                for button in buttons[:3]:
                    try:
                        if button.is_displayed() and button.is_enabled():
                            self.driver.execute_script("arguments[0].click();", button)
                            self._human_sleep(0.2, 0.6)
                    except StaleElementReferenceException:
                        continue
            except Exception:
                continue

    @staticmethod
    def _to_iso_date(date_input):
        """Convert date to YYYY-MM-DD from datetime or common date strings."""
        if isinstance(date_input, datetime):
            return date_input.strftime("%Y-%m-%d")

        for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(date_input, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue

        raise ValueError(f"Unsupported date format: {date_input}")

    def _open_date_picker_if_needed(self):
        """Open date picker if it is not already visible."""
        if self.driver.find_elements(By.CSS_SELECTOR, "div[role='dialog'][aria-label*='calendar'], div[data-testid='searchbox-datepicker']"):
            return

        opener_candidates = [
            (By.CSS_SELECTOR, "button[data-testid='searchbox-dates-container']"),
            (By.CSS_SELECTOR, "button[aria-label*='Check-in date']"),
            (By.XPATH, "//button[contains(@aria-label, 'Check-in')]"),
            (By.XPATH, "//button[contains(@aria-label, 'Select dates')]"),
        ]

        for by, selector in opener_candidates:
            try:
                btn = self.driver.find_element(by, selector)
                if btn.is_displayed() and btn.is_enabled():
                    btn.click()
                    self._human_sleep(0.4, 0.8)
                    return
            except NoSuchElementException:
                continue

    def select_date_range_from_picker(self, checkin_date, checkout_date):
        """
        Select check-in and check-out dates in Booking date picker.

        Args:
            checkin_date: date string or datetime
            checkout_date: date string or datetime
        """
        try:
            checkin_iso = self._to_iso_date(checkin_date)
            checkout_iso = self._to_iso_date(checkout_date)

            in_obj = datetime.strptime(checkin_iso, "%Y-%m-%d")
            out_obj = datetime.strptime(checkout_iso, "%Y-%m-%d")
            if out_obj <= in_obj:
                raise ValueError("check-out date must be after check-in date")

            self._open_date_picker_if_needed()
            wait = WebDriverWait(self.driver, 10)

            checkin_cell = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, f"span[data-date='{checkin_iso}']")
                )
            )
            checkin_cell.click()
            self._human_sleep(0.2, 0.5)

            checkout_cell = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, f"span[data-date='{checkout_iso}']")
                )
            )
            checkout_cell.click()
            self._human_sleep(0.4, 0.9)

            logger.info("Selected date range: %s -> %s", checkin_iso, checkout_iso)
            return True
        except Exception as e:
            logger.error("Failed to select date range: %s", e)
            return False

    @staticmethod
    def _extract_int(text):
        match = re.search(r"\d+", text or "")
        return int(match.group(0)) if match else None

    def _find_occupancy_row(self, popup, label):
        """Find occupancy row by label text (Adults/Children/Rooms)."""
        candidates = popup.find_elements(
            By.XPATH,
            ".//div[.//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '%s')]]"
            % label.lower(),
        )

        if not candidates:
            raise NoSuchElementException(f"Occupancy row not found for {label}")

        # Choose the smallest visible div that has +/- buttons.
        usable = []
        for c in candidates:
            if not c.is_displayed():
                continue
            buttons = c.find_elements(By.XPATH, ".//button[@tabindex='-1']")
            if len(buttons) >= 2:
                area = c.size.get("width", 0) * c.size.get("height", 0)
                usable.append((area, c))

        if not usable:
            raise NoSuchElementException(f"No usable row with +/- for {label}")

        usable.sort(key=lambda x: x[0])
        return usable[0][1]

    def _read_row_value(self, row):
        """Read current occupancy numeric value from a row."""
        buttons = row.find_elements(By.XPATH, ".//button[@tabindex='-1']")
        if len(buttons) >= 2:
            plus_button = buttons[1]
            try:
                value_text = plus_button.find_element(By.XPATH, "preceding-sibling::*[1]").text
                value = self._extract_int(value_text)
                if value is not None:
                    return value
            except Exception:
                pass

            try:
                minus_button = buttons[0]
                value_text = minus_button.find_element(By.XPATH, "following-sibling::*[1]").text
                value = self._extract_int(value_text)
                if value is not None:
                    return value
            except Exception:
                pass

        # Fallback: prefer short numeric fragments in row text.
        numbers = re.findall(r"\b\d{1,2}\b", row.text or "")
        if numbers:
            return int(numbers[-1])

        raise ValueError("Could not read row value")

    def _set_row_value(self, row, target):
        """Adjust row value to target using minus/plus buttons."""
        buttons = row.find_elements(By.XPATH, ".//button[@tabindex='-1']")
        if len(buttons) < 2:
            raise NoSuchElementException("Could not locate +/- buttons")

        minus_btn, plus_btn = buttons[0], buttons[1]
        current = self._read_row_value(row)

        if current == target:
            return

        click_btn = plus_btn if target > current else minus_btn
        steps = abs(target - current)

        for _ in range(steps):
            if click_btn.is_enabled():
                self.driver.execute_script("arguments[0].click();", click_btn)
                self._human_sleep(0.1, 0.25)

    def configure_occupancy(self, adults=2, children=0, rooms=1, pets=False):
        """
        Configure occupancy popup:
        - Adults/Children/Rooms
        - Pets checkbox (name='pets')
        - Child age select(s) (name='age') choose first option
        - Done button
        """
        try:
            wait = WebDriverWait(self.driver, 10)

            config_btn = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-testid='occupancy-config']"))
            )

            aria_expanded = (config_btn.get_attribute("aria-expanded") or "").lower()
            if aria_expanded == "false":
                self.driver.execute_script("arguments[0].click();", config_btn)
                self._human_sleep(0.3, 0.8)

            popup = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='occupancy-popup']"))
            )

            for label, target in (
                ("Adults", adults),
                ("Children", children),
                ("Rooms", rooms),
            ):
                row = self._find_occupancy_row(popup, label)
                self._set_row_value(row, target)
                logger.info("Set %s to %s", label, target)

            # If children > 0, each child has a select name='age'. Select first option.
            if children > 0:
                self._human_sleep(0.3, 0.6)
                age_selects = popup.find_elements(By.CSS_SELECTOR, "select[name='age']")
                for idx, select_elem in enumerate(age_selects[:children]):
                    try:
                        Select(select_elem).select_by_index(0)
                        logger.info("Selected first age option for child %s", idx + 1)
                    except Exception as e:
                        logger.warning("Could not set age for child %s: %s", idx + 1, e)

            # Pets checkbox
            try:
                pets_checkbox = popup.find_element(By.CSS_SELECTOR, "input[name='pets'][type='checkbox']")
                if pets and not pets_checkbox.is_selected():
                    self.driver.execute_script("arguments[0].click();", pets_checkbox)
                if (not pets) and pets_checkbox.is_selected():
                    self.driver.execute_script("arguments[0].click();", pets_checkbox)
            except NoSuchElementException:
                logger.info("Pets checkbox not found")

            done_btn = popup.find_element(
                By.XPATH,
                ".//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'done')]",
            )
            self.driver.execute_script("arguments[0].click();", done_btn)
            self._human_sleep(0.4, 0.9)

            return True
        except Exception as e:
            logger.error("Failed to configure occupancy: %s", e)
            return False

    def search_hotels(
        self,
        city,
        country,
        checkin=None,
        checkout=None,
        adults=2,
        children=0,
        rooms=1,
        pets=False,
    ):
        """Search hotels on Booking homepage."""
        try:
            self.driver.get(self.BASE_URL)
            self._human_sleep(2, 3)
            self._dismiss_popups()

            if checkin is None:
                checkin = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            if checkout is None:
                checkout = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")

            wait = WebDriverWait(self.driver, 12)
            search_xpath = (
                "//input[@name='ss']"
                " | //input[contains(@placeholder, 'Where are you going')]"
                " | //input[contains(@aria-label, 'Where are you going')]"
            )
            search_input = wait.until(
                EC.presence_of_element_located((By.XPATH, search_xpath))
            )

            search_input.clear()
            search_text = f"{city}, {country}"
            for char in search_text:
                search_input.send_keys(char)
                self._human_sleep(0.02, 0.08)

            self._human_sleep(0.7, 1.4)

            # Click first autocomplete option if available.
            suggestion_xpath = "//ul[@role='group']//li[1] | //div[@role='listbox']//div[@role='option'][1]"
            try:
                first_suggestion = wait.until(
                    EC.element_to_be_clickable((By.XPATH, suggestion_xpath))
                )
                first_suggestion.click()
            except TimeoutException:
                logger.warning("No destination suggestion clicked; continuing")

            if not self.select_date_range_from_picker(checkin, checkout):
                return False

            if not self.configure_occupancy(
                adults=adults,
                children=children,
                rooms=rooms,
                pets=pets,
            ):
                return False

            search_btn_xpath = (
                "//button[@type='submit']"
                " | //button[@data-testid='searchbox-submit-button']"
                " | //button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'search')]"
            )
            search_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, search_btn_xpath))
            )
            self.driver.execute_script("arguments[0].click();", search_btn)

            self._human_sleep(4, 6)
            self._dismiss_popups(wait_time=4)
            return True

        except Exception as e:
            logger.error("Error searching hotels: %s", e)
            return False

    def scrape_hotels(self):
        """Scrape hotel cards from search results page."""
        hotels = []

        try:
            wait = WebDriverWait(self.driver, 12)
            wait.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, self.DATA_SELECTORS["card"])
                )
            )

            cards = self.driver.find_elements(By.CSS_SELECTOR, self.DATA_SELECTORS["card"])
            logger.info("Found %s hotel cards", len(cards))

            for idx, card in enumerate(cards):
                try:
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", card)
                    self._human_sleep(0.15, 0.35)

                    hotel = {}

                    try:
                        title_elem = card.find_element(By.CSS_SELECTOR, self.DATA_SELECTORS["title"])
                        hotel["title"] = title_elem.text.strip() or "N/A"
                    except Exception:
                        hotel["title"] = "N/A"

                    try:
                        link_elem = card.find_element(By.CSS_SELECTOR, self.DATA_SELECTORS["link"])
                        hotel["url"] = link_elem.get_attribute("href") or "N/A"
                    except Exception:
                        hotel["url"] = "N/A"

                    try:
                        location_elem = card.find_element(By.CSS_SELECTOR, self.DATA_SELECTORS["location"])
                        hotel["location"] = location_elem.text.strip() or "N/A"
                    except Exception:
                        hotel["location"] = "N/A"

                    try:
                        price_elem = card.find_element(By.CSS_SELECTOR, self.DATA_SELECTORS["price"])
                        hotel["price"] = price_elem.text.strip() or "N/A"
                    except Exception:
                        hotel["price"] = "N/A"

                    try:
                        score_elem = card.find_element(By.CSS_SELECTOR, self.DATA_SELECTORS["review_score"])
                        hotel["review_score"] = score_elem.text.strip() or "N/A"
                    except Exception:
                        hotel["review_score"] = "N/A"

                    try:
                        count_elem = card.find_element(By.CSS_SELECTOR, self.DATA_SELECTORS["review_count"])
                        hotel["review_count"] = count_elem.text.strip() or "N/A"
                    except Exception:
                        hotel["review_count"] = "N/A"

                    hotels.append(hotel)
                    logger.info("Scraped hotel %s: %s", idx + 1, hotel.get("title", "N/A"))

                except StaleElementReferenceException:
                    logger.warning("Stale card at index %s, skipping", idx)
                except Exception as e:
                    logger.error("Error scraping card %s: %s", idx, e)

            return hotels
        except Exception as e:
            logger.error("Error scraping hotels: %s", e)
            return hotels

    def save_to_csv(self, rows, filename):
        """Save scraped rows to CSV."""
        if not rows:
            logger.warning("No rows to save")
            return

        try:
            keys = rows[0].keys()
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

            logger.info("Saved %s rows to %s", len(rows), filename)
        except Exception as e:
            logger.error("Error saving CSV: %s", e)

    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


def main():
    parser = argparse.ArgumentParser(description="Scrape hotels from Booking.com")
    parser.add_argument("--city", required=True, help="City name, e.g. Dubai")
    parser.add_argument("--country", required=True, help="Country name, e.g. United Arab Emirates")

    parser.add_argument(
        "--checkin",
        default=None,
        help="Check-in date (YYYY-MM-DD, MM-DD-YYYY, or DD-MM-YYYY). Default: tomorrow",
    )
    parser.add_argument(
        "--checkout",
        default=None,
        help="Check-out date (YYYY-MM-DD, MM-DD-YYYY, or DD-MM-YYYY). Default: day after tomorrow",
    )

    parser.add_argument("--adults", type=int, default=2, help="Number of adults")
    parser.add_argument("--children", type=int, default=0, help="Number of children")
    parser.add_argument("--rooms", type=int, default=1, help="Number of rooms")
    parser.add_argument("--pets", action="store_true", help="Enable pets filter")

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV filename (default: hotels_{city}_{country}.csv)",
    )
    parser.add_argument(
        "--cookies",
        default="booking/booking_cookies.json",
        help="Path to cookies JSON file",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    args = parser.parse_args()

    if args.adults < 1:
        logger.error("adults must be >= 1")
        return
    if args.children < 0:
        logger.error("children must be >= 0")
        return
    if args.rooms < 1:
        logger.error("rooms must be >= 1")
        return

    output_file = args.output or f"hotels_{args.city}_{args.country}.csv"

    scraper = BookingHotelsScraper(headless=args.headless)

    try:
        if not scraper._load_cookies(args.cookies):
            logger.warning("Continuing without cookies")

        logger.info("Searching hotels in %s, %s", args.city, args.country)
        ok = scraper.search_hotels(
            city=args.city,
            country=args.country,
            checkin=args.checkin,
            checkout=args.checkout,
            adults=args.adults,
            children=args.children,
            rooms=args.rooms,
            pets=args.pets,
        )

        if not ok:
            logger.error("Hotel search failed")
            return

        hotels = scraper.scrape_hotels()
        scraper.save_to_csv(hotels, output_file)
        logger.info("Scraping completed, saved to %s", output_file)

    except Exception as e:
        logger.error("Scraping error: %s", e)
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
