import glassdoor_scraper as gs
import pandas as pd

path = "/usr/lib/chromium-browser/chromedriver"
df = gs.get_jobs("data_scientist", 250, False, path, 15)
df.to_csv("df.csv", index=False)