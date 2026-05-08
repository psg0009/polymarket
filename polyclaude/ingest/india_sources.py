"""India-specific RSS feed list and helper.

The strategy expects ECI/PIB/RBI to outweigh wires which outweigh dailies which
outweigh regional/social — that's enforced by `source_tier` in normalizer.py.
This file just provides the canonical feed bundle so it's all in one place.
"""

from __future__ import annotations

DEFAULT_INDIA_RSS = [
    # Official
    "https://pib.gov.in/AllRel.aspx",  # PIB All releases (HTML; RSS surrogate)
    "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
    # Wires
    "https://www.ptinews.com/feed/",
    # National dailies
    "https://www.thehindu.com/feeder/default.rss",
    "https://indianexpress.com/feed/",
    "https://www.ndtv.com/rss/india",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    # Business
    "https://www.livemint.com/rss/news",
    "https://www.business-standard.com/rss/home_page_top_stories.rss",
]
