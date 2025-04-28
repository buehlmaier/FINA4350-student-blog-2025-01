---
Title: Diving Deep - First Reflections on Journey from the DepthSeeker
Date: 2025-03-24 23:30
Category: Reflective Report
Tags: Group Depthseeker
---

# *First Reflection  - Depthseeker*
Welcome to the DepthSeeker blog, where we document our journey exploring the correlation between social media discussion patterns and cryptocurrency price dynamics.

# *Team Introduction*
Our interdisciplinary team brings diverse perspectives to this project: 

* Anson (Data Science) - Leading our data cleaning and preprocessing, wherein he could utilize his ability to deal with structured data
* Barbie (FinTech) - Helping with the preprocessing of data and taking responsibility for report writing based on her experience in fintech
* Woody (Computer Science) - Heading our web scraping activities and handling data visualization, leveraging his programming expertise
* Apollo (Economics & Finance) - Providing financial analysis expertise and contributing to sentiment analysis, drawing on his economic knowledge

This blend of backgrounds allows us to approach our research from different angles, bringing technical know-how with financial insight. For this particular blog entry, Woody shared his experience with the technical side of data collection, while Barbie helped shape these experiences into a cohesive narrative.

# *Our Project Motivation: Beyond Trading Strategies*
While our first presentation highlighted several practical market applications for our research, including sentiment-driven trading strategies and risk management tools, our ambitions extend far beyond these applications. The group name, "DepthSeeker," captures our true purpose: to dive into the noisy surface of crypto markets and uncover the hidden patterns connecting social sentiment and price movements.

As students with diverse academic backgrounds, we are confronted with the challenge of trying to glean valuable signals from the noisy and sometimes conflicting crypto community. We are thrilled at the prospect of working at the intersection of computer science, behavioral finance, and financial mathematics and discovering areas outside of our respective fields of expertise. In addition to acquiring technical NLP and data analysis abilities, we hope to make academic contributions to cryptocurrency market dynamics research. This project is our learning laboratory where theoretical knowledge meets real-world data problems.

# *The Unexpected Complexity of Web Scraping*
When we first outlined our project timeline, Woody volunteered to carry out the data collection part, which was scraping Reddit and Discord for crypto discussions. What initially seemed easy in theory quickly became a labyrinth of technical challenges.

# *Discord: A Walled Garden*
To perform discord scraping, we need to first find 3 elements

* *USER_TOKEN*: [could be found in developer tools]
* *SERVER_ID*: [after opening the developer mode in discord, right click the server icon]
* *CHANNEL_ID*: [after opening the developer mode in discord, right click the channel icon]

These elements allow us to scrape a specific channel within a specific server, targeting messages before a specified time.

```python
async def scrape_history(channel, before_time, max_batches=5):
   """Scrape historical messages in batches until done or limit reached."""
   total_messages = 0
   batch_count = 0
  
   while batch_count < max_batches:  # Limit batches to avoid over-scraping
       print(f"Batch {batch_count + 1}: Scraping before {before_time}")
       messages_scraped = 0
       with open("crypto_discord_raw.txt", "a", encoding="utf-8") as f:
           async for message in channel.history(limit=1000, before=before_time):
               messages_scraped += 1
               total_messages += 1
               f.write(f"{message.created_at} | {message.author} | {message.content}\n")
               f.flush()
               if messages_scraped % 100 == 0:
                   print(f"Batch {batch_count + 1}: Scraped {messages_scraped} messages (Total: {total_messages})")
               await asyncio.sleep(random.uniform(1, 3))
           # Update before_time to the oldest message in this batch
           if messages_scraped > 0:
               before_time = message.created_at
           else:
               print("No more messages to scrape.")
               break
       print(f"Batch {batch_count + 1} complete. Scraped {messages_scraped} messages.")
       batch_count += 1
       if messages_scraped < 1000:  # Less than limit means we hit the start
           print("Reached the beginning of the channel history.")
           break
       await asyncio.sleep(random.uniform(10, 50))  # Pause between batches to avoid rate limits
  
   print(f"Scraping complete. Total messages scraped: {total_messages}")
```
After the initial setup, I quickly encountered an issue: the current version of the discord.py API is not user-friendly for scraping with a user account. Using a bot for scraping is generally preferred, but since we want to use a user account for simplicity, the latest version poses challenges. To address this, we can use version 1.7.3 of discord.py, which offers easier manipulation for Discord scraping with a user account.

To install this specific version, run the following command in your terminal:

```python
pip install discord.py==1.7.3
```

Woody also discovered a practical limitation with Discord scraping: each request retrieves only about 1,000 messages. This makes selecting a high-quality channel in a reputable server crucial for finding valuable cryptocurrency conversations. Many Discord servers are filled with casual chatter and off-topic noise, diluting crypto-related content. We are still on the hunt for more focused channels that host substantive crypto discussions.

# *Reddit: API Challenge*
To perform Reddit scraping, we first need to create an app within Reddit. This process allows us to obtain the client_id and client_secret, which are essential for initiating the scraping.
Once these credentials are secured, we can begin scraping data, such as the latest posts and comments, using the following approach in our code:

```python
# Fetch the next batch of posts
       submissions = subreddit.new(limit=batch_size, params={'before': last_id} if last_id else {})
       for submission in submissions:
           check_counter += 1  # Increment check counter for every post
           current_time = submission.created_utc
          
           # Print every 100th post being checked
           if check_counter % 100 == 0:
               print(f"Checking post {submission.id}: {datetime.fromtimestamp(current_time)}")
          
           if current_time > end_date:  # Skip posts after Mar 24, 2025
               print(f"  -> Skipping (after end_date)")
               continue
           if current_time < start_date:  # Stop if before Mar 14, 2025
               print(f"  -> Stopping (before start_date)")
               break
          
           post_count += 1
           total_posts += 1
           process_counter += 1  # Increment process counter for every processed post
           batch_posts.append(submission)
          
           # Print every 100th processed post
           if process_counter % 100 == 0:
               print(f"Processing post #{total_posts}: '{submission.title}' (ID: {submission.id}, Time: {datetime.fromtimestamp(current_time)})")
          
           # Store post data
           posts_data.append({
               'post_id': submission.id,
               'title': submission.title,
               'text': submission.selftext,
               'created_utc': datetime.fromtimestamp(current_time),
               'score': submission.score,
               'num_comments': submission.num_comments,
               'url': submission.url
           })
          
           # Fetch all comments
           try:
               submission.comments.replace_more(limit=None)
               comment_count = 0
               for comment in submission.comments.list():
                   comment_count += 1
                   comments_data.append({
                       'post_id': submission.id,
                       'comment_id': comment.id,
                       'body': comment.body,
                       'created_utc': datetime.fromtimestamp(comment.created_utc),
                       'score': comment.score,
                       'parent_id': comment.parent_id
                   })
               print(f"  -> Fetched {comment_count} comments for post {submission.id}")
           except Exception as e:
               print(f"  -> Error fetching comments for post {submission.id}: {e}")
          
           time.sleep(1)  # Respect rate limits
```
Our Reddit data acquisition method hit an unexpected roadblock when Woody uncovered major API changes introduced by Reddit in 2023. Specifically, PRAW (Python Reddit API Wrapper) no longer supports retrieving posts between two specific dates—a feature removed starting with version 6.0.0. Consequently, our current scraping approach is restricted to fetching only the latest posts, as the official API’s free tier no longer supports time-based search queries.
To address this limitation, we’re taking the following steps:

* We are actively exploring alternative methods to access historical Reddit data from specific time periods, which would enhance our historical analysis capabilities.
* In parallel, we are building our own cryptocurrency price dataset, ensuring the dates align with our text data for more cohesive analysis.

# *Cryptocurrency Price Data*
After reviewing datasets available on Kaggle and Hugging Face, Woody determined that most were too outdated for our needs. To overcome this, we turned to direct API calls to Coinbase, which provide historical data for any cryptocurrency we’re interested in, across any timeframe. This approach also opens the door to incorporating real-time pricing into future project implementations.

For example, we can use the following API call to retrieve spot prices with a historical date parameter:
response = requests.get(f'https://api.coinbase.com/v2/prices/{coin_pair}/spot?date={date}')

Note that only the spot price endpoint supports the date parameter for historical price requests.

To illustrate, if we want to fetch BTC/USD pricing data from one year ago, we can structure the request like this:


```python
import requests
import csv
from datetime import datetime, timedelta


# Define the start and end dates for the previous year
end_date = datetime.now().replace(year=datetime.now().year - 1, month=12, day=31)
start_date = end_date.replace(year=end_date.year - 1)


# Define the list of dates to fetch
date_list = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]
coin_pair = 'BTC-USD'
# Prepare CSV file
with open('btc_historical_data.csv', mode='w', newline='') as file:
   writer = csv.writer(file)
   # Write header
   writer.writerow(["Date", "Amount", "Base", "Currency"])
  
   for date in date_list:
       # Make API request for each date
       response = requests.get(f'https://api.coinbase.com/v2/prices/{coin_pair}/spot?date={date}')
       data = response.json()


       # Extract necessary information
       amount = data['data']['amount']
       writer.writerow([date, amount, 'BTC', 'USD'])


print("CSV file has been created successfully.")
```

# *Lesson Learnt*
We have already learned several valuable lessons during this initial stage of the project. What began as a straightforward data collection task evolved into a difficult engineering challenge that considerably pushed our technical abilities forward. We discovered that API documentation literacy is essential, as a thorough reading of platform documentation before implementation would have spared countless hours of debugging and redevelopment. Our Discord experience taught us that specific library versions can distinguish between success and failure, highlighting the importance of dependency management.

We also learned to maximize value from limited computational and financial resources. Strategic data sampling from authoritative sources yielded better results than processing large volumes of low-quality content. These resource optimization techniques proved essential, given our academic project constraints.

These challenges ultimately strengthened our research methodology. Small-scale pilot testing revealed limitations in our approach before we committed further resources to potentially flawed strategies, allowing us to course-correct early and build a more robust analytical framework.

# *Looking Forward*
As we move into the next phase of our project, we will focus on identifying and integrating higher-quality data sources while implementing rigorous data cleansing processes. Our preprocessing pipeline will standardize the dataset by removing irrelevant characters, punctuation, and stop words. We foresee it as particularly challenging owing to the high noise inherent in social media text data. This racket, made up of emojis, platform-specific slang, and intentional misspellings, presents obstacles to correct sentiment analysis. Nevertheless, we recognize that comprehensive cleaning and preprocessing are not merely technical necessities but fundamental prerequisites for coherent signal extraction from the untamed landscape of cryptocurrency discussion online.

Stay tuned for our next blog post, where we will share our experience working on the project!

*(Word Count: 1140)*