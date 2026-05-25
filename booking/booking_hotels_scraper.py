"""
Booking.com Hotels Scraper using Selenium with Edge
Scrapes hotel data from Booking.com search results.

Selectors verified against live property-card HTML (May 2026).
"""

import argparse
import csv
import json
import logging
import math
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
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gauss_clamp(mu, sigma, lo, hi):
    """Gaussian random clamped to [lo, hi]."""
    return max(lo, min(hi, random.gauss(mu, sigma)))


def _bezier_points(x0, y0, x1, y1, steps=20):
    """
    Yield (x, y) points along a quadratic Bézier curve between two screen
    coords.  The control point is offset randomly so the path looks organic.
    """
    cx = (x0 + x1) / 2 + random.randint(-60, 60)
    cy = (y0 + y1) / 2 + random.randint(-60, 60)
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
        yield round(x), round(y)


class BookingHotelsScraper:
    """Scraper for Booking.com hotels — search results page."""

    BASE_URL = "https://www.booking.com/"

    # ── Verified against live property-card HTML (May 2026) ──────────────────
    DATA_SELECTORS = {
        # Container
        "card": "[data-testid='property-card-container']",

        # Title text (the <div> inside the <a>, not the <a> itself)
        "title": "[data-testid='title']",

        # Booking URL
        "link": "a[data-testid='title-link']",

        # Address — lives inside <a data-testid="address-link"> > first <span>
        "location": "[data-testid='address-link'] span:first-child",

        # Discounted (current) price
        "price": "[data-testid='price-and-discounted-price']",

        # Original (struck-through) price — aria-hidden span before the price block
        "original_price": "span[aria-hidden='true'].d68334ea31",

        # Review score numeric value (aria-hidden div inside the score widget)
        "review_score": "[data-testid='review-score'] .f63b14ab7a.dff2e52086",

        # "Very good" / "Excellent" label
        "review_label": "[data-testid='review-score'] .f63b14ab7a.f546354b44",

        # "3,225 reviews"
        "review_count": "[data-testid='review-score'] .fff1944c52.fb14de7f14.eaa8455879",

        # Location sub-score  e.g. "Location 9.4"
        "location_score": "[data-testid='secondary-review-score-link'] span",

        # Distance from centre
        "distance": "button[data-testid='distance']",

        # Star rating container (has aria-label="3 out of 5")
        "stars": "[data-testid='rating-stars']",

        # Deal badge  e.g. "Getaway Deal"
        "deal": "[data-testid='property-card-deal']",

        # Recommended room type
        "room_type": "h4.fff1944c52.f254df5361",

        # "1 night, 2 adults" line
        "stay_info": "[data-testid='price-for-x-nights']",
    }

    def __init__(self, headless=False):
        self.driver = None
        self.headless = headless
        self._last_mouse_x = 400
        self._last_mouse_y = 300
        self._setup_driver()

    # ── Driver setup ──────────────────────────────────────────────────────────

    def _setup_driver(self):
        options = webdriver.EdgeOptions()

        if self.headless:
            options.add_argument("--headless")

        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # Disable images for speed
        prefs = {
            "profile.default_content_settings.images": 2,
            "profile.managed_default_content_settings.images": 2,
        }
        options.add_experimental_option("prefs", prefs)

        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
        )

        self.driver = webdriver.Edge(options=options)
        self.driver.set_window_size(1920, 1080)

        # Hide navigator.webdriver flag
        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
        )
        logger.info("Edge driver initialised")

    # ── Human-behaviour primitives ────────────────────────────────────────────

    def _sleep(self, lo=0.4, hi=1.2):
        """Sleep a Gaussian-distributed amount of time."""
        t = _gauss_clamp((lo + hi) / 2, (hi - lo) / 4, lo, hi)
        time.sleep(t)

    def _micro_sleep(self):
        """Very short pause — used between keystrokes."""
        time.sleep(random.uniform(0.04, 0.18))

    def _move_mouse_to(self, element):
        """
        Move the mouse to *element* along a Bézier curve so the trajectory
        looks organic rather than teleporting straight to the target.
        """
        try:
            rect = self.driver.execute_script(
                "var r=arguments[0].getBoundingClientRect();"
                "return {x:r.left,y:r.top,w:r.width,h:r.height};",
                element,
            )
            # Aim at a random spot inside the element
            target_x = rect["x"] + random.uniform(0.25, 0.75) * rect["w"]
            target_y = rect["y"] + random.uniform(0.25, 0.75) * rect["h"]

            actions = ActionChains(self.driver)
            pts = list(_bezier_points(
                self._last_mouse_x, self._last_mouse_y,
                target_x, target_y,
                steps=random.randint(15, 30),
            ))
            for i, (px, py) in enumerate(pts):
                dx = px - (pts[i - 1][0] if i > 0 else self._last_mouse_x)
                dy = py - (pts[i - 1][1] if i > 0 else self._last_mouse_y)
                actions.move_by_offset(dx, dy)
                # Occasionally pause mid-path
                if random.random() < 0.08:
                    actions.pause(random.uniform(0.05, 0.15))

            actions.perform()
            self._last_mouse_x = target_x
            self._last_mouse_y = target_y
            time.sleep(random.uniform(0.05, 0.15))
        except Exception:
            pass  # Mouse movement is best-effort

    def _human_click(self, element):
        """Move to element then click it."""
        self._move_mouse_to(element)
        try:
            element.click()
        except Exception:
            self.driver.execute_script("arguments[0].click();", element)
        self._sleep(0.2, 0.6)

    def _type_human(self, element, text):
        """
        Type *text* keystroke-by-keystroke with:
        - variable inter-key delays
        - occasional deliberate typo + backspace correction
        - random burst / slow-down phases
        """
        element.clear()
        self._sleep(0.1, 0.3)

        i = 0
        while i < len(text):
            char = text[i]

            # Random typo ~5% of the time (only for alphabetic chars)
            if char.isalpha() and random.random() < 0.05 and i < len(text) - 1:
                wrong = random.choice("qwertyuiopasdfghjklzxcvbnm")
                element.send_keys(wrong)
                time.sleep(random.uniform(0.08, 0.25))
                element.send_keys("\ue003")  # Backspace
                time.sleep(random.uniform(0.1, 0.3))

            element.send_keys(char)

            # Burst mode: fast sequence with occasional pause
            if random.random() < 0.15:
                time.sleep(random.uniform(0.18, 0.55))  # hesitation
            else:
                time.sleep(random.uniform(0.04, 0.14))

            i += 1

        self._sleep(0.1, 0.25)

    def _random_scroll(self, direction="down", pixels=None):
        """Scroll the page a random amount, then pause briefly."""
        if pixels is None:
            pixels = random.randint(200, 600)
        if direction == "up":
            pixels = -pixels
        self.driver.execute_script(f"window.scrollBy(0, {pixels});")
        time.sleep(random.uniform(0.3, 0.8))

    def _idle_wander(self):
        """
        Perform a short idle sequence — scroll a bit, move the mouse — to
        avoid perfectly mechanical timing between page loads.
        """
        steps = random.randint(2, 4)
        for _ in range(steps):
            action = random.choice(["scroll", "mouse", "pause"])
            if action == "scroll":
                self._random_scroll(random.choice(["down", "up"]))
            elif action == "mouse":
                # Move to a random screen location
                x = random.randint(100, 1800)
                y = random.randint(100, 900)
                try:
                    ActionChains(self.driver).move_by_offset(
                        x - self._last_mouse_x, y - self._last_mouse_y
                    ).perform()
                    self._last_mouse_x, self._last_mouse_y = x, y
                except Exception:
                    pass
            else:
                time.sleep(random.uniform(0.4, 1.2))

    # ── Cookie helpers ────────────────────────────────────────────────────────

    def _load_cookies(self, cookies_file):
        try:
            with open(cookies_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)

            self.driver.get(self.BASE_URL)
            self._sleep(0.5, 1.0)

            for cookie in cookies:
                try:
                    c = cookie.copy()
                    for key in ["expirationDate", "sameSite", "storeId"]:
                        c.pop(key, None)
                    if "expiry" not in c and "expirationDate" in cookie:
                        c["expiry"] = int(cookie["expirationDate"])
                    self.driver.add_cookie(c)
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

    # ── Popup dismissal ───────────────────────────────────────────────────────

    def _dismiss_popups(self, wait_time=5):
        selectors = [
            "//button[@aria-label='Dismiss sign in information.']",
            "//button[@aria-label='Dismiss sign-in info.']",
            "//button[@id='onetrust-accept-btn-handler']",
            "//button[@aria-label='Close']",
            "//button[@data-modal-close='true']",
            "//div[@role='dialog']//button[@aria-label='Close']",
            "//button[@aria-label='Dismiss']",
            "//button[contains(@class,'close')]",
            "//button[contains(translate(normalize-space(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept all')]",
        ]
        for sel in selectors:
            try:
                for btn in self.driver.find_elements(By.XPATH, sel)[:3]:
                    try:
                        if btn.is_displayed() and btn.is_enabled():
                            self._human_click(btn)
                    except StaleElementReferenceException:
                        continue
            except Exception:
                continue

    # ── Date helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_iso_date(date_input):
        if isinstance(date_input, datetime):
            return date_input.strftime("%Y-%m-%d")
        for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(date_input, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        raise ValueError(f"Unsupported date format: {date_input}")

    def _open_date_picker_if_needed(self):
        if self.driver.find_elements(
            By.CSS_SELECTOR,
            "div[role='dialog'][aria-label*='calendar'], "
            "div[data-testid='searchbox-datepicker']",
        ):
            return

        for by, sel in [
            (By.CSS_SELECTOR, "button[data-testid='searchbox-dates-container']"),
            (By.CSS_SELECTOR, "button[aria-label*='Check-in date']"),
            (By.XPATH, "//button[contains(@aria-label,'Check-in')]"),
            (By.XPATH, "//button[contains(@aria-label,'Select dates')]"),
        ]:
            try:
                btn = self.driver.find_element(by, sel)
                if btn.is_displayed() and btn.is_enabled():
                    self._human_click(btn)
                    return
            except NoSuchElementException:
                continue

    def select_date_range_from_picker(self, checkin_date, checkout_date):
        try:
            ci = self._to_iso_date(checkin_date)
            co = self._to_iso_date(checkout_date)
            if datetime.strptime(co, "%Y-%m-%d") <= datetime.strptime(ci, "%Y-%m-%d"):
                raise ValueError("check-out date must be after check-in date")

            self._open_date_picker_if_needed()
            wait = WebDriverWait(self.driver, 10)

            self._human_click(
                wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, f"span[data-date='{ci}']")
                ))
            )
            self._sleep(0.2, 0.5)

            self._human_click(
                wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, f"span[data-date='{co}']")
                ))
            )
            self._sleep(0.3, 0.7)

            logger.info("Selected dates: %s → %s", ci, co)
            return True
        except Exception as e:
            logger.error("Failed to select date range: %s", e)
            return False

    # ── Occupancy ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_int(text):
        m = re.search(r"\d+", text or "")
        return int(m.group(0)) if m else None

    def _find_occupancy_row(self, popup, label):
        label_lower = label.lower()
        candidates = popup.find_elements(
            By.XPATH,
            f".//div[.//*[contains(translate(normalize-space(),"
            f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'{label_lower}')]]",
        )
        usable = []
        for c in candidates:
            if not c.is_displayed():
                continue
            buttons = c.find_elements(By.XPATH, ".//button[@tabindex='-1']")
            if len(buttons) >= 2:
                area = c.size.get("width", 0) * c.size.get("height", 0)
                usable.append((area, c))
        if not usable:
            raise NoSuchElementException(f"No occupancy row for {label}")
        usable.sort(key=lambda x: x[0])
        return usable[0][1]

    def _read_row_value(self, row):
        buttons = row.find_elements(By.XPATH, ".//button[@tabindex='-1']")
        if len(buttons) >= 2:
            for sibling_xpath, btn_idx in [
                ("preceding-sibling::*[1]", 1),
                ("following-sibling::*[1]", 0),
            ]:
                try:
                    v = self._extract_int(
                        buttons[btn_idx].find_element(By.XPATH, sibling_xpath).text
                    )
                    if v is not None:
                        return v
                except Exception:
                    pass
        nums = re.findall(r"\b\d{1,2}\b", row.text or "")
        if nums:
            return int(nums[-1])
        raise ValueError("Could not read row value")

    def _set_row_value(self, row, target):
        buttons = row.find_elements(By.XPATH, ".//button[@tabindex='-1']")
        if len(buttons) < 2:
            raise NoSuchElementException("Could not locate +/- buttons")
        minus_btn, plus_btn = buttons[0], buttons[1]
        current = self._read_row_value(row)
        if current == target:
            return
        btn = plus_btn if target > current else minus_btn
        for _ in range(abs(target - current)):
            if btn.is_enabled():
                self._human_click(btn)

    def configure_occupancy(self, adults=2, children=0, rooms=1, pets=False):
        try:
            wait = WebDriverWait(self.driver, 10)
            config_btn = wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "button[data-testid='occupancy-config']")
                )
            )
            if (config_btn.get_attribute("aria-expanded") or "").lower() == "false":
                self._human_click(config_btn)

            popup = wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[data-testid='occupancy-popup']")
                )
            )

            for label, target in (("Adults", adults), ("Children", children), ("Rooms", rooms)):
                self._set_row_value(self._find_occupancy_row(popup, label), target)
                logger.info("Set %s to %s", label, target)

            if children > 0:
                self._sleep(0.3, 0.6)
                for idx, sel in enumerate(
                    popup.find_elements(By.CSS_SELECTOR, "select[name='age']")[:children]
                ):
                    try:
                        Select(sel).select_by_index(0)
                        logger.info("Set age for child %s", idx + 1)
                    except Exception as e:
                        logger.warning("Child %s age: %s", idx + 1, e)

            try:
                cb = popup.find_element(By.CSS_SELECTOR, "input[name='pets'][type='checkbox']")
                if pets != cb.is_selected():
                    self._human_click(cb)
            except NoSuchElementException:
                pass

            done = popup.find_element(
                By.XPATH,
                ".//button[contains(translate(normalize-space(),"
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'done')]",
            )
            self._human_click(done)
            return True
        except Exception as e:
            logger.error("Failed to configure occupancy: %s", e)
            return False

    # ── Search ────────────────────────────────────────────────────────────────

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
        try:
            self.driver.get(self.BASE_URL)
            self._idle_wander()
            self._dismiss_popups()

            if checkin is None:
                checkin = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            if checkout is None:
                checkout = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")

            wait = WebDriverWait(self.driver, 12)
            search_input = wait.until(
                EC.presence_of_element_located((
                    By.XPATH,
                    "//input[@name='ss']"
                    " | //input[contains(@placeholder,'Where are you going')]"
                    " | //input[contains(@aria-label,'Where are you going')]",
                ))
            )

            self._human_click(search_input)
            self._type_human(search_input, f"{city}, {country}")
            self._sleep(0.4, 0.9)

            # First autocomplete suggestion
            try:
                first = wait.until(EC.element_to_be_clickable((
                    By.XPATH,
                    "//ul[@role='group']//li[1] | //div[@role='listbox']//div[@role='option'][1]",
                )))
                self._sleep(0.1, 0.3)
                self._human_click(first)
            except TimeoutException:
                logger.warning("No destination suggestion; continuing")

            if not self.select_date_range_from_picker(checkin, checkout):
                return False

            if not self.configure_occupancy(adults=adults, children=children, rooms=rooms, pets=pets):
                return False

            search_btn = wait.until(EC.element_to_be_clickable((
                By.XPATH,
                "//button[@type='submit']"
                " | //button[@data-testid='searchbox-submit-button']"
                " | //button[contains(translate(normalize-space(),"
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'search')]",
            )))
            self._sleep(0.2, 0.5)
            self._human_click(search_btn)

            # Wait for results — vary timing realistically
            self._sleep(4.5, 7.0)
            self._dismiss_popups(wait_time=4)
            self._idle_wander()
            return True

        except Exception as e:
            logger.error("Error searching hotels: %s", e)
            return False

    # ── Scrape ────────────────────────────────────────────────────────────────

    def _safe_text(self, card, css, default="N/A"):
        """Return stripped text of first matching element, or *default*."""
        try:
            return card.find_element(By.CSS_SELECTOR, css).text.strip() or default
        except Exception:
            return default

    def _safe_attr(self, card, css, attr, default="N/A"):
        try:
            return card.find_element(By.CSS_SELECTOR, css).get_attribute(attr) or default
        except Exception:
            return default

    def scrape_hotels(self):
        hotels = []
        try:
            wait = WebDriverWait(self.driver, 12)
            wait.until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, self.DATA_SELECTORS["card"])
            ))
            cards = self.driver.find_elements(By.CSS_SELECTOR, self.DATA_SELECTORS["card"])
            logger.info("Found %s hotel cards", len(cards))

            for idx, card in enumerate(cards):
                try:
                    # Scroll into view with a bit of randomness around the element
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'});", card
                    )
                    # Random micro-scroll to avoid identical scroll positions
                    self.driver.execute_script(
                        f"window.scrollBy(0, {random.randint(-40, 40)});"
                    )
                    self._sleep(0.2, 0.6)

                    # Occasionally move the mouse over the card (looks natural)
                    if random.random() < 0.4:
                        self._move_mouse_to(card)

                    s = self.DATA_SELECTORS
                    hotel = {
                        "title":          self._safe_text(card, s["title"]),
                        "url":            self._safe_attr(card, s["link"], "href"),
                        "location":       self._safe_text(card, s["location"]),
                        "price":          self._safe_text(card, s["price"]),
                        "original_price": self._safe_text(card, s["original_price"]),
                        "review_score":   self._safe_text(card, s["review_score"]),
                        "review_label":   self._safe_text(card, s["review_label"]),
                        "review_count":   self._safe_text(card, s["review_count"]),
                        "location_score": self._safe_text(card, s["location_score"]),
                        "distance":       self._safe_text(card, s["distance"]),
                        "stars":          self._safe_attr(card, s["stars"], "aria-label"),
                        "deal":           self._safe_text(card, s["deal"]),
                        "room_type":      self._safe_text(card, s["room_type"]),
                        "stay_info":      self._safe_text(card, s["stay_info"]),
                    }
                    hotels.append(hotel)
                    logger.info("Scraped [%s] %s", idx + 1, hotel["title"])

                    # Occasional longer pause between cards
                    if random.random() < 0.12:
                        self._sleep(1.0, 2.5)

                except StaleElementReferenceException:
                    logger.warning("Stale element at index %s, skipping", idx)
                except Exception as e:
                    logger.error("Error scraping card %s: %s", idx, e)

        except Exception as e:
            logger.error("Error scraping hotels: %s", e)

        return hotels

    # ── Output ────────────────────────────────────────────────────────────────

    def save_to_csv(self, rows, filename):
        if not rows:
            logger.warning("No rows to save")
            return
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Saved %s rows → %s", len(rows), filename)
        except Exception as e:
            logger.error("Error saving CSV: %s", e)

    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape hotels from Booking.com")
    parser.add_argument("--city",     required=True, help="City, e.g. Dubai")
    parser.add_argument("--country",  required=True, help="Country, e.g. United Arab Emirates")
    parser.add_argument("--checkin",  default=None,  help="YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--checkout", default=None,  help="YYYY-MM-DD (default: day after tomorrow)")
    parser.add_argument("--adults",   type=int, default=2)
    parser.add_argument("--children", type=int, default=0)
    parser.add_argument("--rooms",    type=int, default=1)
    parser.add_argument("--pets",     action="store_true")
    parser.add_argument("--output",   default=None, help="CSV filename")
    parser.add_argument("--cookies",  default="booking/booking_cookies.json")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.adults < 1:
        logger.error("adults must be >= 1"); return
    if args.children < 0:
        logger.error("children must be >= 0"); return
    if args.rooms < 1:
        logger.error("rooms must be >= 1"); return

    output_file = args.output or f"hotels_{args.city}_{args.country}.csv"
    scraper = BookingHotelsScraper(headless=args.headless)

    try:
        if not scraper._load_cookies(args.cookies):
            logger.warning("Continuing without cookies")

        logger.info("Searching hotels in %s, %s", args.city, args.country)
        ok = scraper.search_hotels(
            city=args.city, country=args.country,
            checkin=args.checkin, checkout=args.checkout,
            adults=args.adults, children=args.children,
            rooms=args.rooms, pets=args.pets,
        )
        if not ok:
            logger.error("Hotel search failed"); return

        hotels = scraper.scrape_hotels()
        scraper.save_to_csv(hotels, output_file)
        logger.info("Done — saved to %s", output_file)

    except Exception as e:
        logger.error("Scraping error: %s", e)
    finally:
        scraper.close()


if __name__ == "__main__":
    main()