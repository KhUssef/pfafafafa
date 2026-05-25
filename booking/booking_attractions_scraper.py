"""
Booking.com Attractions Scraper using Selenium with Edge
Scrapes attractions data from Booking.com
"""

import json
import time
import random
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BookingAttractionsScraper:
    """Scraper for Booking.com attractions"""
    
    BASE_URL = "https://www.booking.com/attractions/"
    
    # Popup dismissal selectors
    POPUP_SELECTORS = [
        "button[aria-label='Dismiss sign in information.']",
        "button[aria-label='Dismiss sign-in info.']",
        "button[id='onetrust-accept-btn-handler']",
        "button[aria-label='Cookie settings']",
        "button:has-text('Accept All Cookies')",
        "button[aria-label='Close']",
        "button[data-modal-close='true']",
        "div[role='dialog'] button[aria-label='Close']",
        "button[aria-label='Dismiss']",
        "button[class*='close']",
    ]
    
    # Data extraction selectors
    DATA_SELECTORS = {
        "card": "li.css-zp7rdd",
        "title": "[data-testid='card-title'] a",
        "location": ".css-1utx3w7",
        "description": ".css-1usy0qg",
        "reviewScore": "[data-testid='review-score'] .css-35ezg3:first-child",
        "reviewLabel": "[data-testid='review-score'] .css-35ezg3:nth-child(2)",
        "reviewCount": "[data-testid='review-score'] .b99b6ef58f:last-child",
        "pricePrefix": "[data-testid='price'] .css-1a9ajzd",
        "priceAmount": "[data-testid='price'] .css-1iufin4",
        "priceNote": "[data-testid='price'] .css-282vld",
        "availability": ".f546354b44",
        "button": "a.de576f5064",
        "mainImage": ".css-1hh4yru",
    }
    
    def __init__(self, headless=False):
        """Initialize the scraper with Edge browser"""
        self.driver = None
        self.headless = headless
        self._setup_driver()
    
    def _setup_driver(self):
        """Setup Edge driver with optimizations"""
        options = webdriver.EdgeOptions()
        
        if self.headless:
            options.add_argument("--headless")
        
        # Disable media loading for speed
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-extensions")
        
        # Additional performance options
        prefs = {
            "profile.default_content_settings.images": 2,  # Disable images
            "profile.managed_default_content_settings.images": 2,
        }
        options.add_experimental_option("prefs", prefs)
        
        # User agent to mimic real browser
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
        )
        options.add_argument(f"user-agent={user_agent}")
        
        self.driver = webdriver.Edge(options=options)
        self.driver.set_window_size(1920, 1080)
        logger.info("Edge driver initialized")
    
    def _load_cookies(self, cookies_file):
        """Load cookies from JSON file"""
        try:
            with open(cookies_file, 'r') as f:
                cookies = json.load(f)
            
            # Navigate to base URL first
            self.driver.get(self.BASE_URL)
            time.sleep(2)
            
            # Add each cookie
            for cookie in cookies:
                try:
                    # Remove problematic fields
                    cookie_copy = cookie.copy()
                    for key in ['expirationDate', 'sameSite', 'storeId']:
                        cookie_copy.pop(key, None)
                    
                    if 'expiry' not in cookie_copy and 'expirationDate' in cookie:
                        cookie_copy['expiry'] = int(cookie['expirationDate'])
                    
                    self.driver.add_cookie(cookie_copy)
                except Exception as e:
                    logger.warning(f"Could not add cookie {cookie.get('name')}: {e}")
            
            logger.info(f"Loaded {len(cookies)} cookies")
            return True
        except FileNotFoundError:
            logger.error(f"Cookies file not found: {cookies_file}")
            return False
        except Exception as e:
            logger.error(f"Error loading cookies: {e}")
            return False
    
    def _human_sleep(self, min_sleep=0.5, max_sleep=2):
        """Sleep for a random duration to mimic human behavior"""
        sleep_time = random.uniform(min_sleep, max_sleep)
        time.sleep(sleep_time)
    
    def _dismiss_popups(self, wait_time=10):
        """Dismiss all visible popups"""
        wait = WebDriverWait(self.driver, wait_time)
        
        popup_selectors_xpath = [
            "//button[@aria-label='Dismiss sign in information.']",
            "//button[@aria-label='Dismiss sign-in info.']",
            "//button[@id='onetrust-accept-btn-handler']",
            "//button[@aria-label='Close']",
            "//button[@data-modal-close='true']",
            "//div[@role='dialog']//button[@aria-label='Close']",
            "//button[@aria-label='Dismiss']",
            "//button[contains(@class, 'close')]",
        ]
        
        for selector in popup_selectors_xpath:
            try:
                buttons = self.driver.find_elements(By.XPATH, selector)
                for button in buttons[:3]:  # Try first 3 matches
                    try:
                        if button.is_displayed():
                            self.driver.execute_script("arguments[0].click();", button)
                            logger.info(f"Dismissed popup: {selector}")
                            self._human_sleep(0.5, 1)
                    except StaleElementReferenceException:
                        continue
            except Exception as e:
                logger.debug(f"Could not find popup {selector}: {e}")
    def select_date_from_picker(self, target_date):
        """
        Select a date using the calendar dialog interface.

        Args:
            target_date: String in format "YYYY-MM-DD" (e.g., "2026-06-15")
                        or datetime object
        """
        # Parse target date
        if isinstance(target_date, str):
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        else:
            target_date_obj = target_date
            target_date = target_date_obj.strftime("%Y-%m-%d")

        target_year = target_date_obj.year
        target_month = target_date_obj.strftime("%B")  # Full month name
        target_day = str(target_date_obj.day)

        # Click input to open dialog (only if aria-haspopup is not "true")
        date_input_xpath = "//button[contains(@aria-label, 'Select dates')]"

        try:
            date_inputs = self.driver.find_elements(By.XPATH, date_input_xpath)
            if date_inputs:
                date_input = date_inputs[0]
                
                # Check for aria-haspopup attribute
                aria_haspopup = date_input.get_attribute("aria-haspopup")
                
                # Only click if aria-haspopup is not "true"
                if aria_haspopup != "true":
                    date_input.click()
                    self._human_sleep(0.5, 1)
                else:
                    logger.info("Button has aria-haspopup='true' - skipping click, assuming dialog is already open")
                    self._human_sleep(0.5, 1)

                # Wait for dialog to appear
                wait = WebDriverWait(self.driver, 5)
                dialog = wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div[role="dialog"][aria-label="Select dates"]')
                ))

                # Navigate to correct month
                max_attempts = 24  # Prevent infinite loop (2 years max)
                attempts = 0

                while attempts < max_attempts:
                    # Get current month header
                    current_month_header = self.driver.find_element(
                        By.CSS_SELECTOR, 'h3[id^="bui-calendar-month-"]'
                    )
                    current_month_text = current_month_header.text  # e.g., "May 2026"

                    if current_month_text == f"{target_month} {target_year}":
                        # Found correct month, break out
                        break
                    
                    # Compare dates to determine if we need next or previous
                    current_date_obj = datetime.strptime(current_month_text, "%B %Y")

                    if target_date_obj > current_date_obj:
                        # Need to go to next month
                        next_button = self.driver.find_element(
                            By.CSS_SELECTOR, 'button[aria-label="Next month"]'
                        )
                        next_button.click()
                    else:
                        # Need previous month (if available)
                        prev_button = self.driver.find_element(
                            By.CSS_SELECTOR, 'button[aria-label="Previous month"]'
                        )
                        prev_button.click()

                    self._human_sleep(0.3, 0.6)
                    attempts += 1

                if attempts >= max_attempts:
                    logger.warning(f"Could not navigate to {target_month} {target_year}")
                    return False

                # Select the specific day
                date_selector = f'span[data-date="{target_date}"][role="checkbox"]'
                day_element = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, date_selector)
                ))
                day_element.click()

                logger.info(f"Successfully selected date: {target_date}")
                self._human_sleep(1, 2)
                return True

            else:
                logger.warning("Date input not found")
                return False

        except Exception as e:
            logger.error(f"Failed to select date: {e}")
            return False
        
    def search_attractions(self, city, country, date=None):
        """Search for attractions in a city"""
        try:
            # Navigate to attractions page
            url = self.BASE_URL
            self.driver.get(url)
            self._human_sleep(2, 3)
            
            # Dismiss initial popups
            self._dismiss_popups()
            
            # If no date provided, use tomorrow
            if not date:
                date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Find and fill search input
            # Use flexible XPath with proper quoting
            search_xpath = "//input[contains(@placeholder, 'Where are you going')]"
            wait = WebDriverWait(self.driver, 10)
            
            try:
                search_input = wait.until(
                    EC.presence_of_element_located((By.XPATH, search_xpath))
                )
                search_input.clear()
                
                # Type city and country
                search_text = f"{city}, {country}"
                for char in search_text:
                    search_input.send_keys(char)
                    self._human_sleep(0.05, 0.15)
                
                logger.info(f"Entered search text: {search_text}")
                self._human_sleep(1, 2)
                
                # Wait for and click the first suggestion
                suggestion_xpath = "//div[@role='listbox']"
                try:
                    first_suggestion = wait.until(
                        EC.element_to_be_clickable((By.XPATH, suggestion_xpath))
                    )
                    first_suggestion.click()
                    logger.info("Selected suggestion")
                except TimeoutException:
                    logger.warning("No suggestions appeared")
                
                self._human_sleep(1, 2)
                
                # Set date if needed
                self.select_date_from_picker(date)
                
                self._human_sleep(1, 2)
                
                # Click search button
                search_button_xpath = (
                    "//button[contains(text(), 'Search')]"
                    " | //button[@type='submit']"
                    " | //button[contains(@class, 'search')]"
                )
                try:
                    search_btn = wait.until(
                        EC.element_to_be_clickable((By.XPATH, search_button_xpath))
                    )
                    self.driver.execute_script("arguments[0].click();", search_btn)
                    logger.info("Clicked search button")
                except TimeoutException:
                    logger.warning("Search button not found, trying alternative selectors")
                    try:
                        # Try any button that might be visible
                        alt_btns = self.driver.find_elements(By.TAG_NAME, "button")
                        for btn in alt_btns:
                            if btn.is_displayed():
                                self.driver.execute_script("arguments[0].click();", btn)
                                logger.info("Clicked alternative search button")
                                break
                    except:
                        pass
                
                # Wait for results to load
                self._human_sleep(4, 6)
                
                # Dismiss any popups that appeared after search
                self._dismiss_popups(wait_time=5)
                
                return True
            except TimeoutException:
                logger.error("Search input not found")
                return False
            
        except Exception as e:
            logger.error(f"Error searching attractions: {e}")
            return False
    
    def scrape_attractions(self):
        """Scrape attraction data from the page"""
        attractions = []
        
        try:
            wait = WebDriverWait(self.driver, 10)
            
            # Wait for cards to load
            wait.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, self.DATA_SELECTORS["card"])
                )
            )
            
            # Get all cards
            cards = self.driver.find_elements(
                By.CSS_SELECTOR, 
                self.DATA_SELECTORS["card"]
            )
            logger.info(f"Found {len(cards)} attraction cards")
            
            for idx, card in enumerate(cards):
                try:
                    # Scroll card into view
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView(true);", card
                    )
                    self._human_sleep(0.3, 0.7)
                    
                    attraction = {}
                    
                    # Extract title
                    try:
                        title_elem = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["title"]
                        )
                        attraction['title'] = title_elem.text
                        attraction['url'] = title_elem.get_attribute('href')
                    except:
                        attraction['title'] = "N/A"
                        attraction['url'] = "N/A"
                    
                    # Extract location
                    try:
                        location_elem = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["location"]
                        )
                        attraction['location'] = location_elem.text
                    except:
                        attraction['location'] = "N/A"
                    
                    # Extract description
                    try:
                        desc_elem = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["description"]
                        )
                        attraction['description'] = desc_elem.text
                    except:
                        attraction['description'] = "N/A"
                    
                    # Extract review score
                    try:
                        score_elem = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["reviewScore"]
                        )
                        attraction['review_score'] = score_elem.text
                    except:
                        attraction['review_score'] = "N/A"
                    
                    # Extract review label
                    try:
                        label_elem = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["reviewLabel"]
                        )
                        attraction['review_label'] = label_elem.text
                    except:
                        attraction['review_label'] = "N/A"
                    
                    # Extract review count
                    try:
                        count_elem = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["reviewCount"]
                        )
                        attraction['review_count'] = count_elem.text
                    except:
                        attraction['review_count'] = "N/A"
                    
                    # Extract price information
                    try:
                        price_prefix = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["pricePrefix"]
                        ).text
                        price_amount = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["priceAmount"]
                        ).text
                        attraction['price'] = f"{price_prefix} {price_amount}".strip()
                    except:
                        attraction['price'] = "N/A"
                    
                    # Extract availability
                    try:
                        avail_elem = card.find_element(
                            By.CSS_SELECTOR, 
                            self.DATA_SELECTORS["availability"]
                        )
                        attraction['availability'] = avail_elem.text
                    except:
                        attraction['availability'] = "N/A"
                    
                    attractions.append(attraction)
                    logger.info(f"Scraped attraction {idx + 1}: {attraction.get('title', 'N/A')}")
                    
                except StaleElementReferenceException:
                    logger.warning(f"Stale element at index {idx}, skipping")
                    continue
                except Exception as e:
                    logger.error(f"Error scraping attraction {idx}: {e}")
                    continue
            
            return attractions
        
        except Exception as e:
            logger.error(f"Error scraping attractions: {e}")
            return attractions
    
    def save_to_csv(self, attractions, filename):
        """Save scraped attractions to CSV"""
        if not attractions:
            logger.warning("No attractions to save")
            return
        
        try:
            keys = attractions[0].keys()
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(attractions)
            
            logger.info(f"Saved {len(attractions)} attractions to {filename}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape attractions from Booking.com"
    )
    parser.add_argument(
        "--city", 
        required=True, 
        help="City name (e.g., Dubai)"
    )
    parser.add_argument(
        "--country", 
        required=True, 
        help="Country name (e.g., United Arab Emirates)"
    )
    parser.add_argument(
        "--date", 
        help="Date in MM-DD-YYYY format (default: tomorrow)"
    )
    parser.add_argument(
        "--output", 
        default=None,
        help="Output CSV filename (default: attractions_{city}_{country}.csv)"
    )
    parser.add_argument(
        "--cookies", 
        default="booking_cookies.json",
        help="Path to cookies JSON file"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run in headless mode"
    )
    
    args = parser.parse_args()
    
    # Validate date format if provided
    if args.date:
        try:
            datetime.strptime(args.date, "%m-%d-%Y")
        except ValueError:
            logger.error("Date must be in MM-DD-YYYY format")
            return
    
    # Set output filename
    output_file = args.output or f"attractions_{args.city}_{args.country}.csv"
    
    # Initialize scraper
    scraper = BookingAttractionsScraper(headless=args.headless)
    
    try:
        # Load cookies
        if not scraper._load_cookies(args.cookies):
            logger.warning("Continuing without cookies")
        
        # Search for attractions
        logger.info(f"Searching for attractions in {args.city}, {args.country}")
        if scraper.search_attractions(args.city, args.country, args.date):
            # Scrape attractions
            attractions = scraper.scrape_attractions()
            
            # Save results
            scraper.save_to_csv(attractions, output_file)
            logger.info(f"Scraping completed. Data saved to {output_file}")
        else:
            logger.error("Search failed")
    
    except Exception as e:
        logger.error(f"Scraping error: {e}")
    
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
