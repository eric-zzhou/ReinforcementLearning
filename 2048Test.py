from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import numpy as np
import neat
from game import Game
import os

# Creates web driver
PATH = "C:\\Program Files (x86)\\chromedriver.exe"
ser = Service(PATH)
op = webdriver.ChromeOptions()
op.add_argument("--mute-audio")
driver = webdriver.Chrome(service=ser, options=op)
driver.get("https://play2048.co/")

# Gets NEAT stuff
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config.txt")
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Load ML model
with open("best.pickle", "rb") as f:
    winner = pickle.load(f)
net = neat.nn.FeedForwardNetwork.create(winner, config)

# Make game
a_moves = ["LEFT", "UP", "RIGHT", "DOWN"]
a_move_keys = [Keys.ARROW_LEFT, Keys.ARROW_UP, Keys.ARROW_RIGHT, Keys.ARROW_DOWN]

# Initializing local game
g = Game(start_two=False)

# Setting up board
tile_container = WebDriverWait(driver, 5).until(  # waits until page loads main
    EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[3]/div[3]"))
)

driver.execute_script("""
   var l = document.getElementsByClassName('ezoic-ad medrectangle-2 medrectangle-2139 adtester-container adtester-container-139')[0];
   l.parentNode.removeChild(l);
""")

while (True):
    try:
        tiles = tile_container.find_elements(By.CSS_SELECTOR, "div[class*=tile-position]")
        webpage = driver.find_element(By.CSS_SELECTOR, "body")

        grid = np.zeros(shape=(4, 4), dtype='uint16')
        for t in tiles:
            # print(t.get_attribute("outerHTML"))
            # print(t.text)
            temp = t.get_attribute("class").split(" ")
            val = temp[1].split("-")[1]
            coords = temp[2].split("-")
            grid[int(coords[3]) - 1][int(coords[2]) - 1] = int(val)

        g.set_grid(grid)
        g.display()

        output = net.activate(tuple(g.flatten()))
        output_tuples = []
        for i, o in enumerate(output):
            output_tuples.append((o, i))
        output_tuples.sort(reverse=True)
        for o, i in output_tuples:
            # print(f"\t({o}, {i})")
            if g.game_move(i):
                webpage.send_keys(a_move_keys[i])
                print(f"Chosen move: {a_moves[i]}")
                break
        driver.implicitly_wait(0.005)
    except StaleElementReferenceException:
        driver.implicitly_wait(0.005)

    try:
        end = driver.find_element(By.CLASS_NAME, "game-message.game-over")
        print(g.score)
        break
    except NoSuchElementException:
        pass
