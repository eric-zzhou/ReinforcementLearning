from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, JavascriptException, \
    ElementNotInteractableException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import numpy as np
from game import AdversarialGame
from Minimax2048 import get_best_move
import logging
import winsound
import time

logging.basicConfig(level=logging.WARNING)
TIME_RUN = (4 * 60) * 60
ONE_RUN = False

# Creates web driver
PATH = "C:\\Program Files (x86)\\chromedriver.exe"
ser = Service(PATH)
op = webdriver.ChromeOptions()
op.add_argument("--mute-audio")
driver = webdriver.Chrome(service=ser, options=op)
driver.get("https://play2048.co/")
start_time = time.time()

# Make game
a_moves = ["LEFT", "UP", "RIGHT", "DOWN"]
a_move_keys = [Keys.ARROW_LEFT, Keys.ARROW_UP, Keys.ARROW_RIGHT, Keys.ARROW_DOWN]

# Initializing local game
g = AdversarialGame(start_two=False)

# Setting up board
tile_container = WebDriverWait(driver, 5).until(  # waits until page loads main
    EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[3]/div[3]"))
)
new_game = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/a")

while True:
    try:
        driver.execute_script("""
       var l = document.getElementsByClassName('ezoic-ad medrectangle-2 medrectangle-2139 adtester-container adtester-container-139')[0];
       l.parentNode.removeChild(l);
        """)
        break
    except JavascriptException:
        for i in range(5):
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            time.sleep(0.1)

while True:
    try:
        tiles = tile_container.find_elements(By.CSS_SELECTOR, "div[class*=tile-position]")
        webpage = driver.find_element(By.CSS_SELECTOR, "body")

        grid = np.zeros(shape=(4, 4), dtype='uint16')
        for t in tiles:
            temp = t.get_attribute("class").split(" ")
            val = temp[1].split("-")[1]
            coords = temp[2].split("-")
            grid[int(coords[3]) - 1][int(coords[2]) - 1] = int(val)

        g.set_grid(grid)
        logging.info(g.display())

        output = get_best_move(g, 5)
        webpage.send_keys(a_move_keys[output])
        logging.info(f"Chosen move: {a_moves[output]}")
    except StaleElementReferenceException:
        driver.implicitly_wait(0.0005)

    try:
        end = driver.find_element(By.CLASS_NAME, "game-message.game-over")
        logging.info(g.score)
        g.reset_game(start_two=False)
        if ONE_RUN or (time.time() - start_time > 14220):
            break
        time.sleep(2.5)
        new_game.click()
        driver.delete_all_cookies()
    except NoSuchElementException:
        pass

    try:
        cont_button = driver.find_element(By.CLASS_NAME, "keep-playing-button")
        cont_button.click()
        driver.delete_all_cookies()
    except (NoSuchElementException, ElementNotInteractableException):
        pass

while True:
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
    time.sleep(0.025)
