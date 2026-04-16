# Weather Expert Manual (Domain Knowledge)

## Protocol
1. **Coordinate Verification**: Always cross-check the geocoding results for the queried location. If the city is "Ashland", verify the state (KY vs OH) using the lat/lon.
2. **Visual Clues**: When analyzed images, look for storm cells, high-altitude cirrus clouds, or visible precipitation streaks.
3. **Comparison Logic**: If comparing temperatures, a change of >10 degrees is "Significant", >20 is "Extreme".
4. **Trend Analysis**: Focus on pressure drops; falling pressure indicates approaching storm fronts.

# Browser Navigator Manual (Domain Knowledge)

## Protocol
1. **Stealth & Reliability**: Prefer clicking text links over CSS selectors when possible for stability across site updates.
2. **Data Extraction**: Always look for JSON-LD or structured metadata in the page source if direct scraping fails.
3. **Task Completion**: Before reporting a task finished, verify the specific data point requested is present in the final extraction summary.
4. **Recovery**: If a page fails to load, try navigating to the homepage and searching internally before giving up.
