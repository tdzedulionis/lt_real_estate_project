"""Web scraper for the Aruodas.lt real estate website using Selenium."""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium_stealth import stealth
from urllib.parse import unquote
import pandas as pd
import logging
import time
import random
import re
from datetime import datetime
import os
import tempfile
from ..config.settings import SCRAPER_CONFIG

class AruodasScraper:
    """A scraper class for extracting real estate data from Aruodas.lt."""

    def __init__(self, category=SCRAPER_CONFIG['category'], 
                 where=SCRAPER_CONFIG['location'], 
                 page_load_timeout=SCRAPER_CONFIG['timeout']):
        """
        Initialize the scraper with configuration settings.

        Args:
            category (str): Real estate category (e.g., 'butai' for apartments)
            where (str): Location filter (e.g., 'vilniuje' for Vilnius)
            page_load_timeout (int): Page load timeout in seconds
        """
        self.category = category
        self.where = where
        self.base_url = f"{SCRAPER_CONFIG['base_url']}/{self.category}/{where}/puslapis/{{}}/?FOrder=AddDate"
        self.data_list = []
        self.page_load_timeout = page_load_timeout

        # Selenium WebDriver setup
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--remote-debugging-port=9222')
        options.add_argument('--disable-web-security')
        options.add_argument('--no-default-browser-check')
        options.add_argument('--no-first-run')
        options.add_argument('--disable-default-apps')
        options.add_argument('--incognito')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.binary_location = '/usr/bin/google-chrome'
        self.driver = webdriver.Chrome(service=Service(), options=options)
        self.driver.set_page_load_timeout(self.page_load_timeout)

        # Apply selenium-stealth
        stealth(
            self.driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def _accept_cookies(self):
        """Accept cookies on the website."""
        try:
            cookie_banner = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, 'onetrust-accept-btn-handler'))
            )
            cookie_banner.click()
            self.logger.info("Cookies accepted")
        except TimeoutException:
            self.logger.warning("Cookie banner not found or timed out")
        except WebDriverException as e:
            self.logger.error(f"WebDriver error handling cookies: {e}")

    def _listing_links(self, page_url):
        """
        Get all listing links from a page.

        Args:
            page_url (str): The URL of the page to scrape

        Returns:
            list: List of listing URLs
        """
        try:
            self.driver.get(page_url)
            time.sleep(random.uniform(1, SCRAPER_CONFIG['wait_time']))
            listing_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@class, "list-adress-v2")]//a'))
            )
            return [link.get_attribute('href') for link in listing_elements if link.get_attribute('href')]
        except TimeoutException:
            self.logger.error(f"Timeout retrieving listing links from {page_url}")
        except WebDriverException as e:
            self.logger.error(f"WebDriver error: {e}")
        return []

    def _extract_coordinates(self, listing_link):
        """
        Extract coordinates from the listing.

        Args:
            listing_link (str): The URL of the listing

        Returns:
            dict: Dictionary containing latitude and longitude
        """
        data = {'latitude': None, 'longitude': None}
        try:
            map_button = self.driver.find_element(By.XPATH, "//a[span[text()='Kaip nuvykti']]")
            map_link = map_button.get_attribute('href')
            coords = unquote(map_link).split("daddr=(")[-1].split(")")[0]
            data['latitude'], data['longitude'] = map(float, coords.split(","))
        except NoSuchElementException:
            try:
                map_button = self.driver.find_element(By.XPATH, "//a[@class='link-obj-thumb vector-thumb-map']")
                map_link = map_button.get_attribute('href')
                query_param = unquote(map_link).split("query=")[-1]
                data['latitude'], data['longitude'] = map(float, query_param.split(","))
            except NoSuchElementException:
                self.logger.warning(f"No map link found for {listing_link}")
            except (IndexError, ValueError) as e:
                self.logger.error(f"Error parsing coordinates in alternative link: {e}")
        except (IndexError, ValueError) as e:
            self.logger.error(f"Error parsing coordinates: {e}")
        return data

    def _parse_listing(self, listing_link):
        """
        Parse a single listing page.

        Args:
            listing_link (str): The URL of the listing to parse

        Returns:
            dict: Dictionary containing the listing data
        """
        try:
            self.driver.get(listing_link)
            time.sleep(SCRAPER_CONFIG['wait_time'])
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//span[@class="price-eur"]'))
            )

            data = {
                "url": listing_link,
                "scrape_date": datetime.today().strftime('%Y-%m-%d')
            }

            # Extract price
            try:
                price_element = self.driver.find_element(By.XPATH, '//span[@class="price-eur"]')
                data["price"] = int(price_element.text.strip().replace(" ", "").replace("€", ""))
            except NoSuchElementException:
                data["price"] = None
                self.logger.warning(f"Could not parse price for {listing_link}")

            # Extract details
            detail_lists = self.driver.find_elements(By.XPATH, '//dl[@class="obj-details  "]')
            for dl in detail_lists:
                labels = dl.find_elements(By.XPATH, "./dt")
                for label_element in labels:
                    label = label_element.text.replace(":", "").strip()
                    value_element = label_element.find_element(By.XPATH, "./following-sibling::dd")
                    value = value_element.text.strip()
                    if label and value:
                        data[label] = value

            # Extract coordinates and city
            data.update(self._extract_coordinates(listing_link))
            
            try:
                header_text = self.driver.find_element(By.XPATH, '//h1[@class="obj-header-text"]').text
                data['city'] = header_text.split(',')[0].strip()
            except NoSuchElementException:
                data['city'] = None
                self.logger.warning(f"Could not find header for {listing_link}")

            # Extract distances
            try:
                distances_grid = self.driver.find_element(By.CLASS_NAME, 'distances-grid')
                distance_items = distances_grid.find_elements(By.CLASS_NAME, 'distance-item')
                for item in distance_items:
                    category = item.get_attribute('data-category')
                    distance_text = item.find_element(By.CLASS_NAME, 'distance-value').text.strip()
            
                    match = re.search(r'([\d.]+)\s*([km]+)', distance_text, re.IGNORECASE)
                    if match:
                        value_str = match.group(1)
                        unit = match.group(2).lower()
            
                        try:
                            value = float(value_str)
                        except ValueError:
                            data[f'distance_to_{category}'] = None
                            continue
            
                        distance_km = value / 1000.0 if unit == 'm' else value
                        data[f'distance_to_{category}'] = distance_km
                    else:
                        match = re.search(r'\d+', distance_text)
                        if match:
                            value = int(match.group(0))
                            data[f'distance_to_{category}'] = value / 1000.0
                        else:
                            data[f'distance_to_{category}'] = None

            except NoSuchElementException:
                self.logger.warning(f"No distances grid found for {listing_link}")
            except Exception as e:
                self.logger.error(f"Error processing distances: {e}")

            self.logger.info(f"Successfully parsed listing: {listing_link}")
            return data

        except WebDriverException as e:
            self.logger.error(f"WebDriver error parsing {listing_link}: {e}")
            return {}

    def scrape_data(self, max_pages="all"):
        """
        Scrape data from multiple pages.

        Args:
            max_pages (int or "all"): Maximum number of pages to scrape
        """
        self.data_list = []
        page = 1
        successful_scrapes = 0
        failed_scrapes = 0
        previous_links = None
        
        print("\nStarting scraping process...")
        url = self.base_url.format(page)
        self.driver.get(url)
        self._accept_cookies()

        while max_pages == "all" or page <= max_pages:
            url = self.base_url.format(page)
            self.driver.get(url)
            time.sleep(2)  # Ensure page loads
            
            # Check if we've reached the last page by finding the disabled "next page" button
            try:
                disabled_next_button = self.driver.find_elements(By.XPATH, '//a[@class="page-bt-disabled" and text()="»"]')
                if disabled_next_button:
                    self.logger.info("Found disabled next page button - reached the last page")
                    break
            except WebDriverException as e:
                self.logger.debug(f"Error checking for disabled next button: {e}")
            
            # Get current page's listing links
            current_links = self._listing_links(url)

            if not current_links:
                self.logger.info(f"No links found on page {page}")
                break
            
            # One more check to test if the last page is reached (compare set of links in previous page with links in curr page)
            if previous_links and set(current_links) == set(previous_links):
                self.logger.info("Reached the last page")
                break

            total_links = len(current_links)
            print(f"\nFound {total_links} listings on page {page}")
            
            for i, link in enumerate(current_links, 1):
                print(f"\rProcessing listing {i}/{total_links} on page {page}...", end="")
                data = self._parse_listing(link)
                if data:
                    self.data_list.append(data)
                    successful_scrapes += 1
                else:
                    failed_scrapes += 1
            print(f"\nCompleted page {page} - Success: {successful_scrapes}, Failed: {failed_scrapes}")
            previous_links = current_links
            page += 1

            # Apply rate limiting between pages
            time.sleep(SCRAPER_CONFIG['wait_time'])

    def close(self):
        """Close the WebDriver instance."""
        self.driver.quit()
        self.logger.info("WebDriver closed")

    def get_data(self):
        """
        Get the scraped data.

        Returns:
            list: List of dictionaries containing scraped data
        """
        return self.data_list
