In this project a model is trained to assess the sentiment (positive or negative) of a movie review.
The model is trained on reviews from the IMDB dataset using logistic regression.

The trained model is then deployed on an HTML page.
The HTML page allows the user to:
- Type in their own review
- Type in a movie title and fetch a review sample for this movie
- Type in a movie title and display movie information (movie poster, genre, overview, etc.)
- Analyze the sentiment of a given movie review

The review samples and movie information are fetched from the TMDB API.

The model is started using app.py and it communicates with the HTML page using Flask. 

![Positive Sentiment](images/Positive%20Sentiment.png)

![Negative Sentiment](images/Negative%20Sentiment.png)
