The goal of the project was to take a CSV file from Kaggle and, using Pandas, NumPy, Matplotlib, Scikit-learn, and other related libraries, create a Linear Regression model that basically creates a graph comparing actual to predicted prices.

With AI's help in syntax, I began first with cleaning the data and having the notion to combine it all.

The Idea: One Smart Predictor

Looking at all those disparate city data files, I knew developing six separate apps would prove to be far too cumbersome. My overriding feeling then was to create one single, smart brain that can predict the prices across major cities. It means the model learns the differences-like recognizing that a house in Mumbai costs more per square foot than the same house in Chennai-but still uses one set of rules.

To do this, I utilized Pandas to load every single city file and mash them all into one giant master table. What I did right at that point was to add a column indicating the city in every row using the filename to tell which city the house belonged to; that column became one of the key clues the model uses.

Making the Data Usable

The data is never clean. First, I changed the price from those huge, confusing numbers into Lakhs (₹ lakh), which just looks better and is easier to read on the final website. Second, I needed to clean up the outliers. You know, the random, ridiculously expensive mansions or tiny, weirdly priced plots. Those rare houses would throw off a simple linear model, so I cut off the top five percent of the most extreme prices. In this way, the model stays focused and accurate for the 95% of houses people are actually interested in.

Teaching the Model to Read Cities

The main technical challenge here was that the Linear Regression model only understands numbers; it cannot read the words "Delhi" or "Koramangala." I applied One-Hot Encoding to fix this.

That essentially means I created a large set of new columns, one for each unique city and location. When you select a house on the website, my code tells the model, "Okay, this house is in Hyderabad," by putting a 1 in the "Hyderabad" column and a 0 in every other city column. That simple number trick is how the model knows exactly which city's premium price adjustment to apply.

Proving the Prediction Works I didn't want the app to be a black box; on the contrary, I wanted it to be transparent. That is why on the right-hand side of the screen, the proof shows: A Scatter Plot is the honest way to check the model. It compares the actual price to the predicted price on data the model had never seen before. If those dots are close to that diagonal line, the model is spot-on. If they stray far off, the prediction is weaker for that price range. And that R² Score is the "accuracy" number: it tells you what percentage of total price variation is explained successfully by the simple clues I gave to the model: the area, the city, and the location. Final Steps and the Website I used Streamlit to handle the whole website part. It takes in the model and the data, and automatically builds all the interactive components, like sliders and dropdowns. All the code is saved on GitHub, and Streamlit Cloud automatically hosts it for me by reading requirements.txt to install all the necessary libraries. That's the whole thought process: from the very first file merge to website deployment. It was all about cleaning the data and telling a simple model how to read city names!
